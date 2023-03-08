package Praktika2;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;
import java.util.Random;

public class HoldOut {
    /*Aurrebaldintzak:
    1. argumentuan .arff fitxategi baten path-a hartzen da. Fitxategi horren klasea azken atributuan dator.
    2. argumentuan irteerarako emaitzak gordetzeko fitxategi baten path-a ematen da*/
    public static void main(String[] args) {
        try {

            String path, path2 = "";
            System.out.println(System.getProperty("user.dir") + "\n");

            if (args.length==0) {
                path = "C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\2LAB\\2. Praktika Datuak-20230129\\adult.train.arff";
                path2="emaitzakHoldOut.txt";
            } else {
                path = args[0];
                path2 = args[1];
            }

            //Datuak dituen fitxategia kargatu:
            ConverterUtils.DataSource source= new ConverterUtils.DataSource(path);
            Instances data=source.getDataSet();
            //Kasu honetan aurrebaldintzatan jartzen duen moduan klasea azkenengo atributua da
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            //https://www.tabnine.com/code/java/classes/weka.filters.unsupervised.instance.RemovePercentage
            //Datuak randomizatu:
            Randomize r=new Randomize();
            r.setInputFormat(data); //TODO: zer input forma sartuko zaion adierazteko
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
            rp.setInvertSelection(true); //Aurrekoaren kontrako datuak erabiliko dira
            Instances trainData = Filter.useFilter(rData, rp);
            System.out.println("Train instantziak: "+trainData.numInstances());
            System.out.println("Test instantziak: "+testData.numInstances());


            //TODO Nuevo:
            trainData.setClassIndex(data.numAttributes() - 1);
            testData.setClassIndex(data.numAttributes() - 1);

            //NaiveBayes classifier eta honekin datuen ebaluaketa burutu
            NaiveBayes klasifikadore= new NaiveBayes();

            //TODO Nuevo: balio zaigu sortzeko klasifikazio erregelak (inferentzia atala) train datu sortarekin
            klasifikadore.buildClassifier(trainData);

            //Ebaluatzailea: //TODO beste behin pasatzen zaio trainData, evaluatzaileak jakiteko zein klase, zenbat atributu...
            Evaluation evaluator = new Evaluation(trainData);
            //TODO zein klasifikatzailekin zer datu testatu (sailkatu)
            evaluator.evaluateModel(klasifikadore, testData);

            System.out.println(evaluator.toSummaryString("\nResults\n======\n", false));
            ///////////////////Ebaluaketaren emaitzak///////////////////
            double acc=evaluator.pctCorrect(); //Accuracy
            double inc=evaluator.pctIncorrect(); //
            double kappa=evaluator.kappa();
            double mae=evaluator.meanAbsoluteError();
            double rmse=evaluator.rootMeanSquaredError();
            double rae=evaluator.relativeAbsoluteError();
            double rrse=evaluator.rootRelativeSquaredError();
            double confMatrix[][]= evaluator.confusionMatrix();

            Date gaur=new Date();

            System.out.println(gaur);
            System.out.println("Correctly Classified Instances  " + acc);
            System.out.println("Incorrectly Classified Instances  " + inc);
            System.out.println("Kappa statistic  " + kappa);
            System.out.println("Mean absolute error  " + mae);
            System.out.println("Root mean squared error  " + rmse);
            System.out.println("Relative absolute error  " + rae);
            System.out.println("Root relative squared error  " + rrse);

            ///////////////////////////////////////////////////////////

            //Fitxategi batean sartu

            FileWriter file = new FileWriter(path2);
            BufferedWriter buffer = new BufferedWriter(file);

            buffer.append("Probaren data: "+ gaur);
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

            System.out.println(evaluator.toClassDetailsString());
            System.out.println(evaluator.precision(1));
            System.out.println("Accuracy (Correctly classified instances): "+evaluator.pctCorrect());

        } catch (FileNotFoundException e) {
            System.out.println("ERROR1: Fitxategi path-a berrikusi:" + args[0]);
        } catch (IOException e) {
            System.out.println("ERROR2: Fitxategi path-aren datuak berrikusi:" + args[0]);
        } catch (Exception e) {e.printStackTrace();}
    }
}
