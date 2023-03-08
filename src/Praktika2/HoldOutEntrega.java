package Praktika2;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;

public class HoldOutEntrega {
    /*Aurrebaldintzak:
    1. argumentuan .arff fitxategi baten path-a hartzen da. Fitxategi horren klasea azken atributuan dator.
    2. argumentuan irteerarako emaitzak gordetzeko fitxategi baten path-a ematen da*/
    public static void main(String[] args) {
        try {

            if (args.length==0) {
                System.out.println("Sartu zuzen komandoa hurrengo formatua erabiliz: \n" +
                        "java -jar holdOut.jar /path/to/data.arff /path/to/emaitzak.txt");
            } else {
                String path, path2 = "";
                path = args[0];
                path2 = args[1];

                //Datuak dituen fitxategia kargatu:
                ConverterUtils.DataSource source= new ConverterUtils.DataSource(path);
                Instances data=source.getDataSet();

                data.setClassIndex(data.numAttributes() - 1);

                //Datuak randomizatu:
                Randomize r=new Randomize();
                r.setInputFormat(data);
                r.setRandomSeed(42);
                Instances rData=Filter.useFilter(data, r);

                //Test egiteko erabiliko diren datuak eskuratu
                RemovePercentage rp =new RemovePercentage();
                rp.setInputFormat(rData);
                rp.setPercentage(66);
                Instances testData=Filter.useFilter(rData, rp);

                //Train-ak egiteko erabiliko diren datuak hartu (Instantzia gehien dituena train)
                rp.setInputFormat(rData);
                rp.setPercentage(66);
                rp.setInvertSelection(true);
                Instances trainData = Filter.useFilter(rData, rp);

                trainData.setClassIndex(data.numAttributes() - 1);
                testData.setClassIndex(data.numAttributes() - 1);

                //NaiveBayes classifier eta honekin datuen ebaluaketa burutu
                NaiveBayes klasifikadore= new NaiveBayes();
                klasifikadore.buildClassifier(trainData);

                //Ebaluatzailea:
                Evaluation evaluator = new Evaluation(trainData);
                evaluator.evaluateModel(klasifikadore, testData);

                //Fitxategi batean sartu

                FileWriter file = new FileWriter(path2);
                BufferedWriter buffer = new BufferedWriter(file);

                buffer.append("Probaren data: "+ new Date());
                buffer.newLine();
                buffer.append("Datu sorta duen dokumentuaren path-a: "+path);
                buffer.newLine();
                buffer.append("Emaitzak dituen dokumentuaren path-a: "+path2);
                buffer.newLine();

                buffer.append("Nahasmen matrizea: ");
                buffer.newLine();
                buffer.append(evaluator.toMatrixString());
                buffer.newLine();

                buffer.append("Klase minoritarioaren datuak: ");
                buffer.newLine();
                int[] maizt = data.attributeStats(data.classIndex()).nominalCounts;
                int minMaiz = maizt[0];
                int minMaizPos=0;
                for(int i=0; i<maizt.length; i++){
                    if(maizt[i]<minMaiz){
                        minMaizPos=i;
                        minMaiz=maizt[i];
                    }
                }
                buffer.append("           Klase minoritarioa: "+data.classAttribute().value(minMaizPos)+"\n");
                buffer.append("           Precision: "+evaluator.precision(minMaizPos)+"\n");
                buffer.append("           Recall: "+evaluator.recall(minMaizPos)+"\n");
                buffer.append("           F-Score: "+evaluator.fMeasure(minMaizPos)+"\n");

                buffer.append("Metrika guztien laburpena: \n");
                buffer.append(evaluator.toSummaryString());
                buffer.newLine();

                buffer.close();
            }
        } catch (FileNotFoundException e) {
            System.out.println("ERROR1: Fitxategi path-a berrikusi:" + args[0]);
        } catch (IOException e) {
            System.out.println("ERROR2: Fitxategi path-aren datuak berrikusi:" + args[0]);
        } catch (Exception e) {e.printStackTrace();}
    }
}
