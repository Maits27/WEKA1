package Praktika2;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Date;

public class EntregaSHoldOut2 {
    /*  Aurrebaldintzak:
        1. argumentuan: train.arff. Fitxategi horren klasea azken atributuan dator.
        2. argumentuan: dev.arff.
        3. argumentuan, evaluation.txt, irteerarako emaitzak gordetzeko fitxategi baten path-a ematen da*/
    public static void main(String[] args) {
        if(args.length==0){
            System.out.println("    Sartutako komandoaren formatua ez da zuzena.\n  Hurrengoko formatua erabili:\n" +
                    "       java -jar HoldOut2.jar /path/to/train.arff /path/to/test.arff /path/to/emaitzak.txt");
        }else{
            try {
                String path0, path1, path2="";
                path0=args[0]; path1=args[1]; path2=args[2];

                ConverterUtils.DataSource source = new ConverterUtils.DataSource(path0);
                Instances train = source.getDataSet();
                source = new ConverterUtils.DataSource(path1);
                Instances test = source.getDataSet();

                train.setClassIndex(train.numAttributes()-1);
                test.setClassIndex(train.numAttributes()-1);

                Randomize r = new Randomize();
                r.setInputFormat(train);
                r.setRandomSeed(42);
                train= Filter.useFilter(train, r);

                Classifier klasifikadore = new NaiveBayes();
                klasifikadore.buildClassifier(train);

                Evaluation evaluator = new Evaluation(train);
                evaluator.evaluateModel(klasifikadore,test);

                FileWriter file = new FileWriter(path2);
                BufferedWriter bf = new BufferedWriter(file);

                bf.append("Exekuzio data: "+ new Date()+"\n");
                bf.append("Exekuziorako argumentuak: \n" +
                        "Train fitxategiaren path-a: "+path0+"\n"+
                        "Test fitxategiaren path-a: "+path1+"\n");
                bf.append("Ateratako nahasmen matrizea:\n");
                bf.append(evaluator.toMatrixString()+"\n");
                bf.append("Accuracy maila: "+evaluator.pctCorrect());

                bf.close();
            }catch (Exception e){System.out.println(e.toString());}
        }
    }
}
