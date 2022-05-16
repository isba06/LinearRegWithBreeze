package made

import scala.util.control.Breaks._
import breeze.linalg
import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, Vector => BV}
import breeze.stats.mean
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.BLAS.dot
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, MetadataUtils}
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.made.LinearRegressionModel.LinearRegressionModelWriter
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder, Row}
import org.apache.spark.mllib

import scala.collection.mutable.ArrayBuffer


trait HasLR extends Params {
  final val lr: DoubleParam = new DoubleParam(this, "lr", "gd learning rate", ParamValidators.gtEq(0))
  final def getLR: Double = $(lr)
}

trait LinearRegressionParams extends PredictorParams with HasLR with HasMaxIter with HasTol {
  protected def validateAndTransformSchema(schema: StructType, fitting: Boolean, featuresDataType: DataType): StructType =
  {
    super.validateAndTransformSchema(schema, fitting, featuresDataType)
  }
  setDefault(lr -> 1e-4, maxIter -> 1000, tol -> 1e-5)
}

class LinearRegressionGD(override val uid: String)
  extends Regressor[Vector, LinearRegressionGD, LinearRegressionModel]
    with LinearRegressionParams with DefaultParamsWritable with Logging {
  def this() = this(Identifiable.randomUID("linRegGD"))
  def setLR(value: Double): this.type = set(lr, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setTol(value: Double): this.type = set(tol, value)
  def copy(extra: ParamMap): LinearRegressionGD = defaultCopy(extra)
  protected def train(dataset: Dataset[_]): LinearRegressionModel = {
    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    implicit val vector_encoder : Encoder[Vector] = ExpressionEncoder()
    implicit val double_encoder : Encoder[Double] = ExpressionEncoder()
    var coef: BDV[Double] = BDV.ones[Double](numFeatures)
    var intercept: Double = 1.0
    var error: Double = Double.MaxValue
    val vectors: Dataset[(Vector, Double)] = dataset.select(dataset($(featuresCol)).as[Vector], dataset($(labelCol)).as[Double])
    breakable { for (i <- 1 to getMaxIter) {
      val (coefficients_summary, intercept_summary) = vectors.rdd.mapPartitions((data: Iterator[(Vector, Double)]) => {
        val coefficients_summarizer = new MultivariateOnlineSummarizer()
        val intercept_summarizer = new MultivariateOnlineSummarizer()
        data.grouped(1000).foreach((r: Seq[(Vector, Double)]) => {
          val (x_, y_) = r.map(x => (x._1.toArray.to[ArrayBuffer], Array(x._2).to[ArrayBuffer])).reduce((x, y) => {(x._1 ++ y._1, x._2 ++ y._2)})
          val x2 = x_.toArray
          val y2 = y_.toArray
          val X = BDM.create(x2.size / numFeatures, numFeatures, x2, 0, numFeatures, true)
          val Y = BDV(y2)
          var Yhat = (X * coef) + intercept
          val residuals = Y - Yhat
          val c: BDM[Double] = X(::, *) * residuals
          coefficients_summarizer.add(mllib.linalg.Vectors.fromBreeze(mean(c(::, *)).t))
          intercept_summarizer.add(mllib.linalg.Vectors.dense(mean(residuals)))
        })
        Iterator((coefficients_summarizer, intercept_summarizer))
      }).reduce((x, y) => {
        (x._1 merge y._1, x._2 merge y._2)
      })
      error = intercept_summary.mean(0)
      if (error.abs < getTol)
        break
      var dCoeff: BDV[Double] = coefficients_summary.mean.asBreeze.toDenseVector
      dCoeff :*= (-2.0) * getLR
      coef -= dCoeff
      var dInter = (-2.0) * getLR * error
      intercept -= dInter
    } }
    val lr_model = copyValues(new LinearRegressionModel(uid, new DenseVector(coefficients.toArray), intercept))
    lr_model
  }
}

class LinearRegressionModel private[made](val uid: String, val coefficients: Vector, val intercept: Double)
  extends RegressionModel[Vector, LinearRegressionModel]
    with LinearRegressionParams with MLWritable {
  val brz_coef: BV[Double] = coef.asBreeze
  private[made] def this(coef: Vector, intercept: Double) = this(Identifiable.randomUID("linRegGD"), coef.toDense, intercept)
  def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(coefficients, intercept))
  def write: MLWriter = new LinearRegressionModelWriter(this)
  def predict(features: Vector): Double = {
    (features.asBreeze dot brzCoefficients) + intercept
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  class LinearRegressionModelWriter(instance: LinearRegressionModel) extends MLWriter {
    private case class Data(intercept: Double, coefficients: Vector)
    def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.intercept, instance.coefficients)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  class LinearRegressionModelReader extends MLReader[LinearRegressionModel] {
    private val className = classOf[LinearRegressionModel].getName
    def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val data = sparkSession.read.format("parquet").load(new Path(path, "data").toString)
      val Row(intercept: Double, coefficients: Vector) = data.select("intercept", "coefficients").head()
      val model = new LinearRegressionModel(metadata.uid, coefficients, intercept)
      metadata.getAndSetParams(model)
      model
    }
  }
  def read: MLReader[LinearRegressionModel] = new LinearRegressionModelReader
  def load(path: String): LinearRegressionModel = super.load(path)
}
