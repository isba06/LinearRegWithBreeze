package spark

import breeze.linalg.DenseVector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.made.{LinearRegressionGD, LinearRegressionModel}

object LinearRegressionSparkApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    val randomizer = new scala.util.Random(1)
    def randomDouble = randomizer.nextDouble
    val randomRDD: RDD[(Double, Double, Double)] = spark.sparkContext.parallelize(
      Seq.fill(100000){(randomDouble, randomDouble, randomDouble)}
    )
    val df: DataFrame = spark.createDataFrame(randomRDD).toDF("X", "Y", "Z")
      .withColumn("F", lit(1.5) * $"X" + lit(0.3) * $"Y" + lit(-0.7) * $"Z" + lit(10))
    val vec_assembler = new VectorAssembler().setInputCols(Array("X", "Y", "Z")).setOutputCol("features")
    val output = vec_assembler.transform(df)
    val lr_solver = new LinearRegressionGD()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setLR(1e-1)
      .setMaxIter(1000)
      .setTol(1e-5)
    val lr_model = lr_solver.fit(output)
    lr_model.write.overwrite().save("linear_model")
    lr_model.transform(output).show(10)
    println(s"Coef: ${lr_model.coefficients}")
    println(s"intercept: ${lr_model.intercept}")
  }
}
