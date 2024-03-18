from pyspark.sql import SparkSession
import pickle
import numpy as np
from pyspark.sql.functions import sin, cos, acos, col

# spark.sparkContext.getConf().getAll()

if __name__ == "__main__":
    '''
    Use PySpark to calculate the distance between each object on a BlackGEM field CCD with all other objects.
    Make use of the local ~2.2 TB Gaia Table rather than astroquery
    Return source_id, Gaia G mag, and the number of objects within 3, 6, 9, 12 arcsec of each BlackGEM source.
    '''

    spark = SparkSession.builder.master("local[*]") \
            .config("spark.driver.memory", "250G") \
            .config("spark.executor.memory", "250G") \
            .appName("Gaia BlackGEM Neighbors") \
            .getOrCreate()

    pi = np.pi

    # https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.DataFrameReader.csv.html
    # Read in the full Gaia DR3 catalog and convert columns to floats (or define schema on read-in)
    gaia_df0 = spark.read.csv("/lustre/research/kupfer/catalogs/GAIA3/data/GaiaSource_*.csv", comment="#", header=True, samplingRatio=0.1)[["source_id", "ra", "dec", "phot_g_mean_mag"]]
    gaia_df = gaia_df0.withColumn("source_id",col("source_id")).withColumn("ra",col("ra").cast("float")).withColumn("dec",col("dec").cast("float")).withColumn("phot_g_mean_mag", col("phot_g_mean_mag").cast("float"))
    gaia_df = gaia_df.filter(col("phot_g_mean_mag") < 20.0).filter(col("phot_g_mean_mag") > 13.0)

    # Given a field ID, return all files with data on that field
    field_id = '11646'
    with open('field_filemap.pkl', 'rb') as ipfile:
        field_filemap = pickle.load(ipfile)
    files = field_filemap[field_id]

    # Read in one of these files at random (they should all have the same objects anyway).
    random_field_file = "/lustre/work/akosakow/blackgem/" + np.random.choice(files).split(".")[0] + ".parquet"
    bg_df = spark.read.parquet(random_field_file)

    # Join the BlackGEM and Gaia DataFrames together, giving a list of objects within the field.
    df = bg_df.join(gaia_df, "source_id").select("source_id", "ra", "dec", "phot_g_mean_mag")
    df = df.repartition(34760)

    # crossJoin the DataFrame with itself (~ 15000x15000 rows) This can be bad for large dataframes)
    df_cjoin = df.crossJoin(df.select("ra", "dec").withColumnRenamed("ra","ra2").withColumnRenamed("dec","dec2"))
    df_cjoin = df_cjoin.repartition(34760)

    # Create new columns counting the number of objects within 12, 9, 6, 3 arcsec
    # This can be re-written to a single distance column rather than calculating distance for each bin.
    df2 = df_cjoin.withColumn("n12", 3600*180*acos(cos(col("dec")*pi/180)*cos(col("dec2")*pi/180)*cos((col("ra") - col("ra2"))*pi/180) + sin(col("dec")*pi/180)*sin(col("dec2")*pi/180))/pi < 12) \
                  .withColumn("n9", 3600*180*acos(cos(col("dec")*pi/180)*cos(col("dec2")*pi/180)*cos((col("ra") - col("ra2"))*pi/180) + sin(col("dec")*pi/180)*sin(col("dec2")*pi/180))/pi < 9) \
                  .withColumn("n6", 3600*180*acos(cos(col("dec")*pi/180)*cos(col("dec2")*pi/180)*cos((col("ra") - col("ra2"))*pi/180) + sin(col("dec")*pi/180)*sin(col("dec2")*pi/180))/pi < 6) \
                  .withColumn("n3", 3600*180*acos(cos(col("dec")*pi/180)*cos(col("dec2")*pi/180)*cos((col("ra") - col("ra2"))*pi/180) + sin(col("dec")*pi/180)*sin(col("dec2")*pi/180))/pi < 3)

    # Spark can't aggregate Booleans, so recast the True/False into 1/0. 
    df3 = df2.withColumn("source_id", col("source_id")).withColumn("n12",col("n12").cast("integer")) \
                                     .withColumn("n9",col("n9").cast("integer")) \
                                     .withColumn("n6",col("n6").cast("integer")) \
                                     .withColumn("n3",col("n3").cast("integer")) \
             .withColumn("phot_g_mean_mag", col("phot_g_mean_mag"))

    # Aggregate-sum each n-column by source_id.
    # The result is the "number of objects within radius of source_id" with 1 being the main object.
    df4 = df3.groupBy("source_id", "phot_g_mean_mag").agg({"n12":"sum", "n9":"sum", "n6":"sum", "n3":"sum"})

    # Coalesce into one partition to create a single output file of ~10000 output rows. This can be slow/impossible for large outputs depending on driver memory
    df4.coalesce(1).write.option("header",True).csv("output")

    spark.stop()
