package Praktika5;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class AtributuHautapen2 {
    /*ARGUMENTUAK:
    *   1. data.arff: gainbegiratutako instantzien path (input)
        2. NB.model: eredua gordetzeko irteerako path (output)*/
    public static void main(String[] args) {
        try {
            String dataPath, modelPath, selekPath ;
            if (args.length == 0) {
                System.out.println("Sartutako komando egitura ez da zuzena. Hurrengoko eredua jarraitu:\n" +
                        "java -jar FSSetaNB.jar /path/to/data.arff /path/to/karpeta/NB.model /path/to/datuenSelekzio.arff");
                dataPath = "E:\\EHES\\WEKAPRUEBAS\\adult.train.arff";
                modelPath = "E:\\EHES\\WEKAPRUEBAS\\nb.model";
                selekPath = "E:\\EHES\\WEKAPRUEBAS\\datuenSelekzio.arff";
            } else {
                dataPath = args[0];
                modelPath = args[1];
                selekPath = args[2];
            }
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            //TODO BUILD EVALUATOR ----> SI NO ES UN FILTRO
            AttributeSelection as = new AttributeSelection();
            as.setEvaluator(new CfsSubsetEval());
            as.setSearch(new BestFirst());
            as.setInputFormat(data);
            Instances selection = Filter.useFilter(data, as);
            selection.setClassIndex(selection.numAttributes()-1);

            DataSink dataSink= new ConverterUtils.DataSink(selekPath);
            dataSink.write(selection);

            System.out.println(data.numAttributes());
            for (int i=0; i<data.numAttributes(); i++){
                System.out.println(data.attribute(i).name());
            }

            System.out.println(selection.numAttributes());
            for (int i=0; i<selection.numAttributes(); i++){
                System.out.println(selection.attribute(i).name());
            }

            Classifier nb = new NaiveBayes();
            nb.buildClassifier(selection);
            System.out.println(selection.instance(1));
            weka.core.SerializationHelper.write(modelPath, nb);

        }catch (Exception e){System.out.println(e.toString());}
    }
}
