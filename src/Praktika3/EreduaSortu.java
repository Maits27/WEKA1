package Praktika3;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Date;
import java.util.Random;

public class EreduaSortu {
    /* Aurrebaldintzen parametroak:
    1. data.arff: datuen path (input)
    2. NB.model: eredua gordetzeko path (output)
    3. KalitatearenEstimazioa.txt: kalitatearen estimazioa gordetzeko path (output)*/
    //MODELOA GORDETZEKO:
    //https://stackoverflow.com/questions/33556543/how-to-save-model-and-apply-it-on-a-test-dataset-on-java
    public static void main(String[] args) {
        if(args.length==0){
            System.out.println("    Sartutako komandoaren formatoa ez da zuzena. \n     Hurrengo formatua duen agindua sartu:\n" +
                    "       java -jar Praktika3.EreduaSortu.jar /path/to/data.arff /path/to/modeloa.model /path/to/KalitateEstimazioa.txt");
        }else{
            try {
                String path, path1, path2 = "";
                path = args[0]; path1 = args[1]; path2 = args[2];

                ConverterUtils.DataSource source= new ConverterUtils.DataSource(path);
                Instances data=source.getDataSet();
                data.setClassIndex(data.numAttributes() - 1);

                //Randomize filter
                Randomize r=new Randomize();
                r.setInputFormat(data);
                r.setRandomSeed(42);
                Instances rData= Filter.useFilter(data, r);

                //K-FCV
                Evaluation evaluator = new Evaluation(rData);
                evaluator.crossValidateModel(new NaiveBayes(), rData, 10, new Random(42));
                System.out.println(evaluator.toClassDetailsString());

                //Hold-Out 70
                //Train
                RemovePercentage rp = new RemovePercentage();
                rp.setInputFormat(rData);
                rp.setPercentage(70);
                rp.setInvertSelection(true);
                Instances train = Filter.useFilter(rData, rp);
                train.setClassIndex(train.numAttributes()-1);

                //Test
                rp.setInputFormat(rData);
                rp.setPercentage(70);
                Instances test = Filter.useFilter(rData, rp);
                test.setClassIndex(test.numAttributes()-1);

                //Modeloa entrenatu:
                Classifier klasifikadore= new NaiveBayes();
                klasifikadore.buildClassifier(train);

                //Modeloaren kalitate estimazioa:
                Evaluation evaluator2 = new Evaluation(train);
                evaluator2.evaluateModel(klasifikadore, test);

                FileWriter file=new FileWriter(path2);
                BufferedWriter bf=new BufferedWriter(file);

                //MODELOA GORDE:
                klasifikadore.buildClassifier(rData); //TODO MODELOA ENTRENATZEKO GUZTIAREKIN EGIN BEHAR DA GORDETZERAKO ORDUAN
                weka.core.SerializationHelper.write(path1, klasifikadore);

                //Kalitatearen estimazioa eman:
                bf.append(new Date().toString());
                bf.newLine(); bf.newLine();
                bf.append("Datuen path-a: "+path);
                bf.newLine(); bf.newLine();
                bf.append(".model aurkitzeko path-a: "+path1);
                bf.newLine(); bf.newLine();
                bf.append("K-FOLD CROSS VALIDATION EGITERAKOAN LORTUTAKO NAHASMEN MATRIZEA: ");
                bf.newLine();
                bf.append(evaluator.toMatrixString());
                bf.newLine();
                bf.append("%70 PROPORTZIOKO HOLD-OUT EGITERAKOAN LORTUTAKO NAHASMEN MATRIZEA: ");
                bf.newLine();
                bf.append(evaluator2.toMatrixString());
                bf.close();
            }catch (Exception e){System.out.println(e.toString());}
        }
    }
}
