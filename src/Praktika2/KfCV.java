package Praktika2;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.io.*;
import java.util.Date;

/*Va a dar errores de las librerias de WEKA en el comando filter.setInputFormat(data);
* Para evitarlo:
* Clicar encima del nombre de la clase en la parte arriba a la derecha (al lado del play)
* Edit configurations
* Modify options
* Add VM options
* En el hueco que sale: --add-opens java.base/java.lang=ALL-UNNAMED*/

public class KfCV {
    /*Aurrebaldintzak:
    1. argumentuan .arff fitxategi baten path-a hartzen da. Fitxategi horren klasea azken atributuan dator.
    2. argumentuan irteerarako emaitzak gordetzeko fitxategi baten path-a ematen da*/
    public static void main(String[] args) {
        try {

            String path, path2 = "";
            //System.out.println(System.getProperty("user.dir") + "\n");

            if (args.length==0) {
                System.out.println("Prototipoa: java -jar estimateNaiveBayes5fCV.jar  /path/to/data.arff   /path/to/emaitzak.txt \n" +
                        "Helburua: emandako datuekin Naive Bayes-en kalitatearen estimazioa lortu 5-fCV eskemaren bidez eta datuei buruzko informazioa eman\n" +
                        "Argumentuak:\n" +
                        "1. Datu sortaren kokapena (path) .arff  formatuan (input). Aurre-baldintza: klasea azken atributuan egongo da.\n" +
                        "2. Emaitzak idazteko irteerako fitxategiaren path-a (output).");
            } else {
                path = args[0]; path2 = args[1];
                //Datuak dituen fitxategia kargatu:
                ConverterUtils.DataSource source= new ConverterUtils.DataSource(path);
                Instances data=source.getDataSet();
                //Kasu honetan aurrebaldintzatan jartzen duen moduan klasea azkenengo atributua da
                if (data.classIndex() == -1) {
                    data.setClassIndex(data.numAttributes() - 1);
                }

                //Setting up the filter -> filters the data and obtains the reduced dataset
                //Aukeratutako filtroa (supervised->AtributeSelection)
                AttributeSelection filter = new AttributeSelection();
                // Preprocessing step of attribute selection
                //AtributeSelection barruan (WEKA-n) agertzen diren aukerak:
                CfsSubsetEval eval = new CfsSubsetEval();
                BestFirst search = new BestFirst();
                //Filtroa prestatu:
                filter.setEvaluator(eval);
                filter.setSearch(search);
                filter.setInputFormat(data);
                //Filtroa aplikatu:
                Instances newData = Filter.useFilter(data, filter);
                //Instances newData=data;

                //NaiveBayes classifier eta honekin datuen ebaluaketa burutu
                NaiveBayes klasifikadore= new NaiveBayes();
                Evaluation evaluator = new Evaluation(newData);
                //5-fCV + Random=1 ("no shuffle") WEKA-ko berdina ateratzeko Random bera jarri
                evaluator.crossValidateModel(klasifikadore, newData, 5, new Random(1));

                ///////////////////Ebaluaketaren emaitzak///////////////////
                double acc=evaluator.pctCorrect(); //Accuracy
                double inc=evaluator.pctIncorrect(); //
                double kappa=evaluator.kappa();
                double mae=evaluator.meanAbsoluteError();
                double rmse=evaluator.rootMeanSquaredError();
                double rae=evaluator.relativeAbsoluteError();
                double rrse=evaluator.rootRelativeSquaredError();
                double confMatrix[][]= evaluator.confusionMatrix();

                /*
                System.out.println("Correctly Classified Instances  " + acc);
                System.out.println("Incorrectly Classified Instances  " + inc);
                System.out.println("Kappa statistic  " + kappa);
                System.out.println("Mean absolute error  " + mae);
                System.out.println("Root mean squared error  " + rmse);
                System.out.println("Relative absolute error  " + rae);
                System.out.println("Root relative squared error  " + rrse);
                */
                ///////////////////////////////////////////////////////////

                //Fitxategi batean sartu

                FileWriter file = new FileWriter(path2);
                BufferedWriter buffer = new BufferedWriter(file);

                buffer.append("Probaren data: "+ new Date());
                buffer.newLine();
                buffer.append("Datu sorta duen dokumentuaren path-a: "+path);
                buffer.newLine();
                buffer.append("Datu sorta duen dokumentuaren path-a: "+path2);
                buffer.newLine();
                buffer.append(evaluator.toSummaryString()+"\n"+evaluator.toClassDetailsString());
                buffer.append("Nahasmen matrizea: ");
                String matrize= evaluator.toMatrixString();
                System.out.println(matrize);
                buffer.append(matrize);
                buffer.newLine();

                //TODO precision metrika klasearen balio bakoitzeko eta weighted avg
                String p=evaluator.toClassDetailsString();
                buffer.append("Prezizioa: "+p);
                buffer.close();
                System.out.println(p);
            }


        } catch (FileNotFoundException e) {
                System.out.println("ERROR1: Fitxategi path-a berrikusi:" + args[0]);
            } catch (IOException e) {
                System.out.println("ERROR2: Fitxategi path-aren datuak berrikusi:" + args[0]);
            } catch (Exception e) {e.printStackTrace();}
    }
}
