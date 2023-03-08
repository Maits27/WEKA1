package Praktika5;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class AtributuHautapen2IkerNago {
    public static void main(String[] args) {
        /*ARGUMENTUAK:
            1. train.arff: gainbegiratutako instantzien path (input)
            2. NB.model: eredua gordetzeko irteerako path (output)*/
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
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);

            //1. Eredu iragarle optimoa sortu eta gorde
            AttributeSelection as = new AttributeSelection();
            as.setInputFormat(data);
            Instances selection = Filter.useFilter(data, as);
            selection.setClassIndex(selection.numAttributes()-1);

                //Randomize
                Randomize r = new Randomize();
                r.setInputFormat(data);
                Instances rData = Filter.useFilter(data,r);
                rData.setClassIndex(rData.numAttributes()-1);

                //Split Data
                RemovePercentage rp = new RemovePercentage();
                rp.setPercentage(70);
                rp.setInvertSelection(true);
                rp.setInputFormat(data);
                Instances train = Filter.useFilter(rData, rp);
                train.setClassIndex(train.numAttributes()-1);

                rp.setPercentage(70);
                rp.setInvertSelection(false);
                rp.setInputFormat(data);
                Instances test = Filter.useFilter(rData, rp);
                test.setClassIndex(test.numAttributes()-1);


                //Sailkatzailea entrenatu
                Classifier nb = new NaiveBayes();
                nb.buildClassifier(train);

                //Ebaluatu
                Evaluation holdOut = new Evaluation(train);
                holdOut.evaluateModel(nb,test);

                System.out.println("Atributu kopurua: "+selection.numAttributes());
                System.out.println("\n"+holdOut.toMatrixString()+"\n");
                System.out.println("F-score: "+holdOut.weightedFMeasure()+"\n");

            nb = new NaiveBayes();
            nb.buildClassifier(selection);
            weka.core.SerializationHelper.write(modelPath,nb);

            //2. Test multzoaren iragarpenak egin

        }catch (Exception e){System.out.println(e.toString());}
    }
}
