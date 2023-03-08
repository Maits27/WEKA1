package Praktika2;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Date;

public class StratifiedHoldOut2 {
    /*  Aurrebaldintzak:
        1. argumentuan: train.arff. Fitxategi horren klasea azken atributuan dator.
        2. argumentuan: dev.arff.
        3. argumentuan, evaluation.txt, irteerarako emaitzak gordetzeko fitxategi baten path-a ematen da*/
    public static void main(String[] args) {
        try {
            String path0, path1, path2="";
            if(args.length==0){
                path0 = "C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\2LAB\\2. Praktika Datuak-20230129\\strain2.arff";
                path1 = "C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\2LAB\\2. Praktika Datuak-20230129\\stest2.arff";
                path2 = "StratifiedEvaluation2.txt";
            }else{
                path0=args[0]; path1=args[1]; path2=args[2];
            }
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
