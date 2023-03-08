package Praktika2;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class hold_out_Aingeru {
    public static void main(String[] args) {
        if(args.length == 0){
            System.out.println(
                    "Programa honek bi argumentu behar diru:\n\n" +
                            "  1. Datu fitxategien path-a.\n\n" +
                            "  2. Emaitzak gordetzeko fitxategiaren path-a.\n\n" +
                            "Gainera, Java-ren 'InaccessibleObjectException' gainditzeko, aukera bat jarri behar da aukeren artean.\n\n" +
                            "Sintaxia hurrengoa izango litzateke:\n\n" +
                            "     java -jar ---add-opens java.base/java.lang=ALL-UNNAMED '<jar_path>' <arg0> <arg1>\n\n"
            );
        }
        else {
            try {
                // ******* 1. LOAD DATA **********************************************************************
                DataSource source = new DataSource(args[0]);
                Instances data = source.getDataSet();
                data.setClassIndex(data.numAttributes() - 1);


                // ******* 2. FILTER aplikatu 'train' eta 'test' lortzeko a************************************
                // 2.1. RANDOMIZE
                Randomize filter_random = new Randomize();
                filter_random.setInputFormat(data);
                Filter.useFilter(data, filter_random);

                // 2.2. REMOVE PERCENTAGE
                RemovePercentage filter = new RemovePercentage();
                filter.setPercentage(66);
                filter.setInputFormat(data);

                Instances test_instances = Filter.useFilter(data, filter);

                filter.setInvertSelection(true); //eta ez setPercentage(36);
                filter.setInputFormat(data);
                Instances train_instances = Filter.useFilter(data, filter);


                // ******* 3. EVALUATZAILEA **********************************
                NaiveBayes estimador = new NaiveBayes(); //sailkatzailea (classifier)
                estimador.buildClassifier(train_instances); //hau erabili behar da, ez delako k-fCV

                Evaluation evaluator = new Evaluation(train_instances);
                evaluator.evaluateModel(estimador, test_instances);


                // ******* 4. EMAITZAK **********************************************
                // System.out.println(evaluator.toSummaryString());
                SimpleDateFormat fromateador = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
                Date date = new Date();
                String actaulDate = fromateador.format(date);
                String confMat = evaluator.toMatrixString(); //evaluator.confusionMatrix();

                try{
                    FileWriter myWriter = new FileWriter(args[1]);
                    myWriter.write(actaulDate + "\n\n" +
                            "Datuen fitxategia: " + args[0] + "\n\n" +
                            "Emaitzak gordetzeko fitxategia: " + args[1] + "\n\n");
                    myWriter.write(confMat + "\n");
                    myWriter.write("=== Precission ===\n\n");
                    System.out.println(data.classAttribute().getLowerNumericBound());
                    for(int i=0;i<data.classAttribute().numValues();i++){
                        myWriter.write(data.classAttribute().value(i) + " --> " +  evaluator.precision(i) + "\n");
                    }
                    myWriter.write("Weighted average: " + evaluator.weightedFMeasure());
                    myWriter.close();
                }
                catch (IOException e){
                    System.out.println("Errore bat egon da fitxategiarekin");
                    e.printStackTrace();
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
