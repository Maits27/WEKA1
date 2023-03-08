package Praktika5;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.misc.InputMappedClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;
import weka.filters.unsupervised.instance.Resample;

import java.io.BufferedWriter;
import java.io.FileWriter;

public class Atala4NireMetodokin {
    public static void main(String[] args) {
        try {
            String testPath, modelPath, dataPath, emaPath, trainPath, blindPath;
            if (args.length == 0) {
                System.out.println("Sartutako komando egitura ez da zuzena. Hurrengoko eredua jarraitu:\n" +
                        "java -jar Exekutagarri.jar /path/to/data.arff /path/to/train.arff /path/to/test.arff " +
                        "/path/to/blind_test.arff /path/to/karpeta/filtered.model /path/to/ema.txt");
                dataPath = "E:\\EHES\\WEKAPRUEBAS\\data_supervised.arff";
                trainPath ="E:\\EHES\\WEKAPRUEBAS\\70train3.arff";
                testPath = "E:\\EHES\\WEKAPRUEBAS\\70test3.arff";
                blindPath = "E:\\EHES\\WEKAPRUEBAS\\blind70test3.arff";
                modelPath = "E:\\EHES\\WEKAPRUEBAS\\filtered3.model";
                emaPath = "E:\\EHES\\WEKAPRUEBAS\\emaitza3FiltratutakoAtributukin5.txt";
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
            Instances selectedTrain = Filter.useFilter(train, attributeSelection);
            selectedTrain.setClassIndex(selectedTrain.numAttributes()-1);
            System.out.println("Aukeratutako atributu kopurua: "+selectedTrain.numAttributes());

            //Klasifikadorea
            Classifier nb = new NaiveBayes();
            nb.buildClassifier(selectedTrain);
            InputMappedClassifier fk = new InputMappedClassifier();
            fk.setModelHeader(selectedTrain);
            fk.setClassifier(nb);
            fk.setSuppressMappingReport(true);

            /*Classifier fk = new NaiveBayes();
            fk.buildClassifier(train);

            /*FilteredClassifier fk = new FilteredClassifier();
            fk.setClassifier(new NaiveBayes());
            fk.buildClassifier(selectedTrain);

            /*AttributeSelectedClassifier fk = new AttributeSelectedClassifier();
            fk.setClassifier(new NaiveBayes());
            fk.setEvaluator(new CfsSubsetEval());
            fk.setSearch(new BestFirst());
            fk.buildClassifier(selectedTrain);*/

            //Gorde eta jaso klasifikadorea:
            weka.core.SerializationHelper.write(modelPath, fk);
            Classifier k = (Classifier) weka.core.SerializationHelper.read(modelPath);
            //Classifier k = (FilteredClassifier) weka.core.SerializationHelper.read(modelPath);
            //Classifier k = (AttributeSelectedClassifier) weka.core.SerializationHelper.read(modelPath);

            //Gorde egindako partiketa
            ConverterUtils.DataSink ds = new ConverterUtils.DataSink(trainPath);
            ds.write(selectedTrain);
            ds = new ConverterUtils.DataSink(testPath);
            ds.write(test);
            ds = new ConverterUtils.DataSink(blindPath);
            ds.write(blind_test);

            //Kendu atributu gehigarriak testari:
            int atrib[]=new int[selectedTrain.numAttributes()];
            for(int i =0; i<selectedTrain.numAttributes(); i++){
                for (int j =0; j<test.numAttributes(); j++){
                    if(selectedTrain.attribute(i).name().equalsIgnoreCase(test.attribute(j).name())){
                        atrib[i]=j;
                    }
                }
            }

            Remove r = new Remove();
            r.setAttributeIndicesArray(atrib);
            r.setInvertSelection(true);
            r.setInputFormat(test);
            Instances selectedTest = Filter.useFilter(test, r);
            selectedTest.setClassIndex(selectedTest.numAttributes()-1);

            r.setAttributeIndicesArray(atrib);
            r.setInvertSelection(true);
            r.setInputFormat(test);
            Instances selectedBlindTest = Filter.useFilter(blind_test, r);
            selectedBlindTest.setClassIndex(selectedBlindTest.numAttributes()-1);

            //Ebaluatu:
            Evaluation evaluation = new Evaluation(selectedTrain);
            evaluation.evaluateModel(k, test);

            //Idatzi ebaluazioa + Klasifikazioa:
            FileWriter f = new FileWriter(emaPath);
            BufferedWriter bf = new BufferedWriter(f);
            bf.append("EBALUAZIOA:\n");
            bf.append(evaluation.toSummaryString());
            bf.append(evaluation.toClassDetailsString());
            bf.append(evaluation.toMatrixString());
            bf.newLine();
            bf.append("\nKLASIFIKAZIOA:\n");

            for (int i = 0; i<blind_test.numInstances(); i++){
                bf.append(i+" instantzia ---> "+k.classifyInstance(blind_test.instance(i))+"\n");
            }

            bf.close();

        }catch (Exception e){System.out.println(e.toString());}
    }
}
