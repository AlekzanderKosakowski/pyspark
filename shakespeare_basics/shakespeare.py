# Intro to PySpark basic code from "Machine Learning and Big Data, January 29&31, 2024"

# spark-submit filename.py
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.html

import re
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("shakespeare")
sc = SparkContext(conf=conf)

if __name__ == "__main__":

    # Read in a text file as an RDD
    rdd = sc.textFile("Complete_Shakespeare.txt")

    # Flatten the RDD into 1D RDD with one "word" per element
    rdd_flat = rdd.flatMap(lambda line: line.lower().split())

    # Remove punctuation
    rdd_clean = rdd_flat.map(lambda ele: re.sub("[^A-Za-z0-9 ]+","",ele))

    # Create key-value pairs for all elements in the RDD. Key=word, Value=count
    rdd_kv = rdd_clean.map(lambda ele: (ele, 1))

    # Reduce the RDD into an RDD with unique keys, adding the values associated together (count the number of words)
    rdd_count = rdd_kv.reduceByKey(lambda x,y: x+y)

    # Flip the key and values to allow sorting by value (count)
    rdd_count2 = rdd_count.map(lambda x: (x[1], x[0]))

    # Sort the RDD by the count (key)
    rdd_sort = rdd_count2.sortByKey(False)

    # Show the top 5 most common words
    rdd_sort.take(5)

    # Save the RDD
    rdd_sort.saveAsTextFile("outputs")

