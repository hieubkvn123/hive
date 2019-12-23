package hive;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;


public class SVMOnHiveDS{
    private static final String APP_NAME = "AHIHI";
    private static final String MASTER = "local[*]";

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

            return RowFactory.create(loanID, gender, married, education, applicantIncome, loanAmount, creditHistory, outcome);
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

        Dataset<Row> updated_ds = spark.createDataFrame(rdd, schema);
        Dataset<Row>[] splits = updated_ds.randomSplit(new double[]{0.9,0.1});
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        String[] col_list = new String[]{"gender", "married","education","applicantIncome","loanAmount","creditHistory"};
        VectorAssembler va = new VectorAssembler().setInputCols(col_list).setOutputCol("features");
        LinearSVC svc = new LinearSVC().setMaxIter(10).setRegParam(0.01).setFeaturesCol("features").setPredictionCol("prediction").setLabelCol("outcome");

        Pipeline p = new Pipeline().setStages(new PipelineStage[]{va, svc});
        PipelineModel pModel = p.fit(train);
        Dataset<Row> results = pModel.transform(test);

        System.out.println("Prediction results : ");
        results.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
                .setPredictionCol("prediction")
                .setLabelCol("outcome");

        double accuracy = evaluator.evaluate(results);
        System.out.println("Accuracy = " + accuracy);

        results.filter("outcome == 1").select("loanID").coalesce(1).write().format("csv").csv("/Users/yenmm/Desktop/risky");
        results.filter("outcome == 0").select("loanID").coalesce(1).write().format("csv").csv("/Users/yenmm/Desktop/notRisky");

        long risky_count = results.filter("outcome == 1").count();
        long unrisky_count = results.filter("outcome == 0").count();

        JFrame chart = new RiskBarChart("Risk bar chart", risky_count, unrisky_count);
        chart.setSize(800, 400);
        chart.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        chart.setVisible(true);
    }

    public static class RiskBarChart extends JFrame{
      public RiskBarChart(String title, long risky, long not_risky){
        super(title);
        CategoryDataset dataset = createDataset(risky, not_risky);

        JFreeChart chart = ChartFactory.createBarChart(
          "Risky vs Non-risky customers", // Chart title
          "Riskiness of borrowers",       // X-axis label
          "Number of borrowers",          // Y-axis label
          dataset,                        // The dataset
          PlotOrientation.VERTICAL,       // Plot orientation
          true, true, false               // legends - tooltips - urls
        );

        ChartPanel chartPanel = new ChartPanel(chart);
        setContentPane(chartPanel);
      }
      public CategoryDataset createDataset(long risky, long not_risky){
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.addValue(risky, "Number of clients", "Risky");
        dataset.addValue(not_risky, "Number of clients", "Not risky");

        return dataset;
      }
    }
}
