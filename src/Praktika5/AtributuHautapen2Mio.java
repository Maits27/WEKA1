package Praktika5;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;

public class AtributuHautapen2Mio {
    /*ARGUMENTUAK:
    *   1. train.arff: gainbegiratutako instantzien path (input)
        2. NB.model: eredua gordetzeko irteerako path (output)*/
    public static void main(String[] args) {
        try {
            String trainPath, modelPath;
            if (args.length == 0) {
                System.out.println("Sartutako komando egitura ez da zuzena. Hurrengoko eredua jarraitu:\n" +
                        "java -jar FSSetaNB.jar /path/to/train.arff /path/to/karpeta/NB.model");
                trainPath = "E:\\EHES\\WEKAPRUEBAS\\70train.arff";
                modelPath = "E:\\EHES\\WEKAPRUEBAS\\nb.model";
            } else {
                trainPath = args[0];
                modelPath = args[1];
            }
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(trainPath);
            Instances train = source.getDataSet();
            train.setClassIndex(train.numAttributes()-1);

            boolean[] geratu = new boolean[train.numAttributes()-1];
            for (int i=0; i< geratu.length; i++){geratu[i]=false;}

            for(int i=0; i< train.numAttributes()-1; i++){
                int[] x = new int[2];
                x[0]=i; x[1]=train.numAttributes()-1;

                Remove r = new Remove();
                r.setInvertSelection(true);
                r.setAttributeIndicesArray(x);
                r.setInputFormat(train);
                Instances aux = Filter.useFilter(train, r);
                aux.setClassIndex(aux.numAttributes()-1);

                Classifier nb = new NaiveBayes();
                nb.buildClassifier(aux);
                Evaluation evaluation= new Evaluation(aux);
                evaluation.crossValidateModel(nb, aux, 5, new Random());

                if(evaluation.fMeasure(Utils.minIndex(train.attributeStats(train.numAttributes()-1).nominalCounts))>0){geratu[i]=true;}

                System.out.println(aux.numAttributes() + " " +aux.attribute(0).name()+ " fm: " +evaluation.fMeasure(Utils.minIndex(train.attributeStats(train.numAttributes()-1).nominalCounts)));
            }
            int kont=0;
            for(int i=0; i<geratu.length;i++){if(!geratu[i]){kont++;}}
            int[] x = new int[kont];
            kont=0;
            for(int i=0; i<geratu.length;i++){if(!geratu[i]){x[kont]=i;kont++;}}

            Remove r = new Remove();
            r.setAttributeIndicesArray(x);
            r.setInvertSelection(false);
            r.setInputFormat(train);
            Instances ftrain = Filter.useFilter(train, r);
            ftrain.setClassIndex(ftrain.numAttributes()-1);

            Classifier nb = new NaiveBayes();
            nb.buildClassifier(ftrain);

            System.out.println(ftrain.numAttributes());
            weka.core.SerializationHelper.write(modelPath, nb);
        }catch (Exception e){System.out.println(e.toString());}
    }
}
