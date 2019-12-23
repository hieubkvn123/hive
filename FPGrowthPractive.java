package hive;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.api.java.JavaRDD;

import java.util.List;
import java.util.Arrays;

import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.mllib.fpm.AssociationRules;

public class FPGrowthPractive {
    private static final String APP_NAME = "ahihi";
    private static final String MASTER = "local[*]";

    public static void main(String[] args){
        SparkConf conf = new SparkConf().setAppName(APP_NAME).setMaster(MASTER);
        JavaSparkContext context = new JavaSparkContext(conf);

        String data_file = "/Users/yenmm/Desktop/fpm.txt";
        JavaRDD<String> data = context.textFile(data_file);

        JavaRDD<List<String>> transactions = data.map(s -> {
            return Arrays.asList(s.split(" "));
        });

        FPGrowth fpGrowth = new FPGrowth().setMinSupport(0.5);
        FPGrowthModel<String> model = fpGrowth.run(transactions);

        for(FPGrowth.FreqItemset<String> item : model.freqItemsets().toJavaRDD().collect()){
            System.out.printf("%s\t%d%n", item.javaItems(), item.freq());
        }

        for(AssociationRules.Rule<String> rule : model.generateAssociationRules(0.5).toJavaRDD().collect()){
            System.out.printf("%s\t%s\t%.2f\n", rule.javaAntecedent(), rule.javaConsequent(), rule.confidence());
        }

    }
}
