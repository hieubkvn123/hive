package hive;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.api.java.JavaRDD;

// NOTE : Go to hive-site.xml set hive.metastore.uris to thrift://localhost:9083
// NOTE : Set the spark.sql.warehouse.dir to /user/hive/warehouse (hdfs location)
// NOTE : export JAVA_HOME=/path/to/java_8/home

public class RandomForestOnHiveDS{
    private static final String APP_NAME = "AHIHI";
    private static final String MASTER =  "local[*]";

    public static void main(String[] args){
        SparkSession spark =  SparkSession.builder().master(MASTER).appName(APP_NAME)
                .config("spark.sql.warehouse.dir", "/user/hive/warehouse")
                .config("hive.metastore.uris", "thrift://localhost:9083")
                .enableHiveSupport()
                .getOrCreate();
        spark.sql("USE default");
        Dataset<Row> bank_risk = spark.sql("SELECT * FROM bank_risk");
        Dataset<Row> non_na_bank_risk = bank_risk.na().drop();

        System.out.println("Full dataset : ");
        non_na_bank_risk.show();

        // Mapping the categorical data into numeric
        JavaRDD<Row> rdd = non_na_bank_risk.javaRDD().map(s -> {
          int gender = 0;
          int education = 0;
          int married = 0;
          int outcome = 0;
          double applicantIncome = 0;
          double loanAmount = 0;
          int creditHistory = 0;

          if(s.get(1).toString().equals("Male"))
            gender = 1;
          if(s.get(2).toString().equals("Yes"))
            married = 1;
          if(s.get(4).toString().equals("Graduate"))
            education = 1;
          if(s.get(12).toString().equals("Y"))
            outcome = 1;

          applicantIncome = Double.parseDouble(s.get(6).toString());
          loanAmount = Double.parseDouble(s.get(8).toString());
          creditHistory = Integer.parseInt(s.get(10).toString());

          return RowFactory.create(
              s.get(0).toString(),gender, education, married, applicantIncome, loanAmount, creditHistory, outcome
          );

        });

        StructType schema = new StructType(new StructField[] {
          new StructField("loanID", DataTypes.StringType, false, Metadata.empty()),
          new StructField("gender", DataTypes.IntegerType, false, Metadata.empty()),
          new StructField("married", DataTypes.IntegerType, false, Metadata.empty()),
          new StructField("education", DataTypes.IntegerType, false, Metadata.empty()),
          new StructField("applicantIncome", DataTypes.DoubleType, false, Metadata.empty()),
          new StructField("loanAmount", DataTypes.DoubleType, false, Metadata.empty()),
          new StructField("creditHistory", DataTypes.IntegerType, false, Metadata.empty()),
          new StructField("outcome", DataTypes.IntegerType, false, Metadata.empty())
        });

        Dataset<Row> updated_ds = spark.createDataFrame(rdd, schema);
        Dataset<Row>[] splits = updated_ds.randomSplit(new double[]{0.9,0.1});
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        String[] col_list = new String[]{"gender","married","education","applicantIncome","loanAmount","creditHistory"};
        VectorAssembler va = new VectorAssembler().setInputCols(col_list).setOutputCol("features");
        RandomForestClassifier rf_classifier = new RandomForestClassifier().setFeaturesCol("features")
                                                                           .setLabelCol("outcome")
                                                                           .setPredictionCol("prediction");


        Pipeline p = new Pipeline().setStages(new PipelineStage[]{va, rf_classifier});
        PipelineModel pModel = p.fit(train);

        Dataset<Row> results = pModel.transform(test);
        System.out.println("Prediction Results : ");
        results.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("outcome")
                                                                                             .setPredictionCol("prediction")
                                                                                             .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(results);
        System.out.println("Accuracy = " + accuracy);

        results.filter("outcome == 0").select("loanID").coalesce(1).write().format("csv").csv("/Users/yenmm/Desktop/notRisky");
        results.filter("outcome == 1").select("loanID").coalesce(1).write().format("csv").csv("/Users/yenmm/Desktop/risky");
    }
}
