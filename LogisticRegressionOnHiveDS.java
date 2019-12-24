package hive;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.Metadata;

import org.apache.spark.api.java.JavaRDD;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import javax.swing.JFrame;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.category.CategoryDataset;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.plot.PlotOrientation;

public class LogisticRegressionOnHiveDS {
  private static final String APP_NAME = "AHIHI";
  private static final String MASTER   = "local[*]";

  public static void main(String[] args){
    SparkSession spark = SparkSession.builder().appName(APP_NAME).master(MASTER)
                                        .config("hive.metastore.uris", "thrift://localhost:9083")
                                        .config("spark.sql.warehouse.dir", "/user/hive/warehouse")
                                        .enableHiveSupport()
                                        .getOrCreate();

    spark.sql("USE default");
    Dataset<Row> raw_ds = spark.sql("SELECT * FROM bank_risk");
    Dataset<Row> ds = raw_ds.na().drop();

    JavaRDD<Row> rdd = ds.javaRDD().map(s -> {
      int gender = 0;
      int married = 0;
      int education = 0;
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
      String loanID = s.get(0).toString();

      return RowFactory.create(
          loanID, gender, married, education, applicantIncome, loanAmount, creditHistory, outcome
      );
    });

    StructType schema = new StructType(new StructField[]{
      new StructField("loanID", DataTypes.StringType, false, Metadata.empty()),
      new StructField("gender", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("married", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("education", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("applicantIncome", DataTypes.DoubleType, false, Metadata.empty()),
      new StructField("loanAmount", DataTypes.DoubleType, false, Metadata.empty()),
      new StructField("creditHistory", DataTypes.IntegerType, false, Metadata.empty()),
      new StructField("outcome", DataTypes.IntegerType, false, Metadata.empty())
    });

    Dataset<Row> dataset = spark.createDataFrame(rdd, schema);
    Dataset<Row>[] splits = dataset.randomSplit(new double[]{0.9,0.1});
    Dataset<Row> train  = splits[0];
    Dataset<Row> test   = splits[1];

    String[] col_list = new String[]{"gender","married","education","applicantIncome","loanAmount","creditHistory"};
    VectorAssembler va = new VectorAssembler().setInputCols(col_list).setOutputCol("features");
    LogisticRegression lr = new LogisticRegression().setRegParam(0.01).setMaxIter(1000)
                                                    .setFeaturesCol("features")
                                                    .setLabelCol("outcome")
                                                    .setPredictionCol("prediction");

    Pipeline p = new Pipeline().setStages(new PipelineStage[]{va, lr});
    PipelineModel pModel = p.fit(train);
    Dataset<Row> results = pModel.transform(test);

    System.out.println("Prediction results : " );
    results.show();

    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
                                                                                         .setLabelCol("outcome")
                                                                                         .setPredictionCol("prediction");

    double accuracy = evaluator.evaluate(results);
    System.out.println("Accuracy =  " + accuracy);

    results.filter("outcome == 1").select("loanID").coalesce(1).write().mode("overwrite").format("csv").csv("/Users/yenmm/Desktop/notRisky");
    results.filter("outcome == 0").select("loanID").coalesce(1).write().mode("overwrite").format("csv").csv("/Users/yenmm/Desktop/risky");

    long risky_count = results.filter("outcome == 0").count();
    long notRisky_count = results.filter("outcome == 1").count();

    JFrame frame = new RiskBarChart("Number of Risky vs Not-Risky customers - LogisticRegression", risky_count, notRisky_count);
    frame.setSize(300, 900);
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setVisible(true);
  }

  public static class RiskBarChart extends JFrame {
    public RiskBarChart(String title, long risky, long notRisky){
      super(title);
      // TODO : chart goes here
      CategoryDataset dataset = createDataset(risky, notRisky);
      JFreeChart chart = ChartFactory.createBarChart(
        "Number of Risky vs Not-Risky customers", // Chart title
        "Categories",                             // X-axis label
        "Number of borrowers",                    // Y-axis label
        dataset,                                  // dataset
        PlotOrientation.VERTICAL,                 // plot orientation
        true, true, false                         // legend - tooltips - urls
      );

      ChartPanel chartPanel = new ChartPanel(chart);
      setContentPane(chartPanel);
    }

    public CategoryDataset createDataset(long risky, long notRisky){
      DefaultCategoryDataset dataset = new DefaultCategoryDataset();
      dataset.addValue(risky, "Number of borrowers", "Risky");
      dataset.addValue(notRisky, "Number of borrowers", "Not risky");

      return dataset;
    }
  }
}
