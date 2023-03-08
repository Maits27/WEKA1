package Praktika5;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.Resample;

import java.io.BufferedWriter;
import java.io.FileWriter;

public class Atala4 {
    public static void main(String[] args) {
        try {
            String testPath, modelPath, dataPath, emaPath, trainPath, blindPath;
            if (args.length == 0) {
                System.out.println("Sartutako komando egitura ez da zuzena. Hurrengoko eredua jarraitu:\n" +
                        "java -jar Exekutagarri.jar /path/to/data.arff /path/to/train.arff /path/to/test.arff " +
                        "/path/to/blind_test.arff /path/to/karpeta/filtered.model /path/to/ema.txt");
                dataPath = "E:\\EHES\\WEKAPRUEBAS\\data_supervised.arff";
                trainPath ="E:\\EHES\\WEKAPRUEBAS\\70train2.arff";
                testPath = "E:\\EHES\\WEKAPRUEBAS\\70test2.arff";
                blindPath = "E:\\EHES\\WEKAPRUEBAS\\blind70test2.arff";
                modelPath = "E:\\EHES\\WEKAPRUEBAS\\filtered.model";
                emaPath = "E:\\EHES\\WEKAPRUEBAS\\emaitzaFiltratutakoAtributukin5.txt";
            } else {
                dataPath = args[0];
                trainPath=args[1];
                testPath = args[2];
                blindPath =args[3];
                modelPath = args[4];
                emaPath = args[5];
            }
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);

            //Stratified hold out:
            Resample resample = new Resample();
            resample.setRandomSeed(42);
            resample.setInvertSelection(false);
            resample.setNoReplacement(true);
            resample.setSampleSizePercent(70);
            resample.setInputFormat(data);
            Instances train = Filter.useFilter(data, resample);
            train.setClassIndex(train.numAttributes()-1);

            resample.setRandomSeed(42);
            resample.setInvertSelection(true);
            resample.setNoReplacement(true);
            resample.setSampleSizePercent(70);
            resample.setInputFormat(data);
            Instances test = Filter.useFilter(data, resample);
            test.setClassIndex(test.numAttributes()-1);

            //Egin blind test
            ReplaceWithMissingValue rpwmv = new ReplaceWithMissingValue();
            rpwmv.setProbability(1);
            rpwmv.setIgnoreClass(false);
            rpwmv.setAttributeIndicesArray(new int[]{test.classIndex()});
            rpwmv.setInputFormat(test);
            Instances blind_test = Filter.useFilter(test, rpwmv);
            blind_test.setClassIndex(blind_test.numAttributes()-1);

            //Aukeratu atributu hoberenak
            AttributeSelection attributeSelection=new AttributeSelection();
            attributeSelection.setEvaluator(new CfsSubsetEval());
            attributeSelection.setSearch(new BestFirst());
            attributeSelection.setInputFormat(train);
            /*Instances selectedTrain = Filter.useFilter(train, attributeSelection);
            selectedTrain.setClassIndex(selectedTrain.numAttributes()-1);
            System.out.println("Aukeratutako atributu kopurua: "+selectedTrain.numAttributes());*/

            //Klasifikadorea
            FilteredClassifier fk2 = new FilteredClassifier();
            fk2.setClassifier(new NaiveBayes());
            fk2.setFilter(attributeSelection);
            fk2.buildClassifier(train);

            AttributeSelectedClassifier fk = new AttributeSelectedClassifier();
            fk.setClassifier(new NaiveBayes());
            fk.setEvaluator(new CfsSubsetEval());
            fk.setSearch(new BestFirst());
            fk.buildClassifier(train);

            //Gorde eta jaso klasifikadorea:
            //weka.core.SerializationHelper.write(modelPath, fk);
            //Classifier k = (FilteredClassifier) weka.core.SerializationHelper.read(modelPath);
            //Classifier k = (AttributeSelectedClassifier) weka.core.SerializationHelper.read(modelPath);

            /*Gorde egindako partiketa
            ConverterUtils.DataSink ds = new ConverterUtils.DataSink(trainPath);
            ds.write(selectedTrain);
            ds = new ConverterUtils.DataSink(testPath);
            ds.write(test);
            ds = new ConverterUtils.DataSink(blindPath);
            ds.write(blind_test);*/

            //Ebaluatu:
            Evaluation evaluation = new Evaluation(train);
            evaluation.evaluateModel(fk, test);

            FileWriter f = new FileWriter(emaPath);
            BufferedWriter bf = new BufferedWriter(f);

            //Idatzi ebaluazioa + Klasifikazioa:
            bf.append("EBALUAZIOA:\n");
            bf.append(evaluation.toSummaryString());
            bf.append(evaluation.toClassDetailsString());
            bf.append(evaluation.toMatrixString());
            bf.newLine();
            bf.append("\nKLASIFIKAZIOA:\n");

            for (int i = 0; i<blind_test.numInstances(); i++){
                bf.append(i+" instantzia ---> "+fk.classifyInstance(blind_test.instance(i))+"\n");
                //Filtered classifier bitartez ere:
                System.out.println(i+" instantzia ---> "+fk2.classifyInstance(blind_test.instance(i)));
            }

            bf.close();

        }catch (Exception e){System.out.println(e.toString());}
    }
}
