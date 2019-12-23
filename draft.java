package  hive;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SQLContext;

import org.apache.spark.api.java.JavaRDD;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.mllib.fpm.AssociationRules;

public class draft
{
    private static final String APP_NAME = "AHIHI";
    private static final String MASTER = "local[*]";

    public static void main(String[] args) throws Exception{
        SparkConf conf = new SparkConf().setAppName(APP_NAME).setMaster(MASTER);
        JavaSparkContext context = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(context);

        JavaRDD<String> lines = context.textFile("/Users/yenmm/Desktop/fpm.txt");
        JavaRDD<List<String>> transactions = lines.map(s -> {
            return Arrays.asList(s.split(" "));
        });

        FPGrowth fpGrowth = new FPGrowth().setMinSupport(0.5);
        FPGrowthModel<String> model = fpGrowth.run(transactions);


        for (FPGrowth.FreqItemset<String> item : model.freqItemsets().toJavaRDD().collect()){
            System.out.printf("%s\t%d%n", item.javaItems(), item.freq());
        }

        List<AssociationRules.Rule<String>> rules = model.generateAssociationRules(0.8).toJavaRDD().collect();
        for (AssociationRules.Rule<String> rule : rules){
            System.out.printf("%s\t%s\t%.2f\n", rule.javaAntecedent(), rule.javaConsequent(), rule.confidence());
        }
    }

}