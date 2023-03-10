package AZTERKETA2;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;
import weka.filters.unsupervised.instance.Resample;

public class RForest {
    public static void main(String[] args) throws Exception{
        String dataPath, testPath, trainPath, blindPath, modelPath, emaPath;
        if(args.length==0){
            System.out.println("Ez da formato zuzena sartu.");
            dataPath="E:\\EHES\\WEKAPRUEBAS\\neutrons.arff";
            trainPath ="E:\\EHES\\WEKAPRUEBAS\\a2train.arff";
            testPath ="E:\\EHES\\WEKAPRUEBAS\\a2test.arff";
            blindPath ="E:\\EHES\\WEKAPRUEBAS\\a2blindtest.arff";
            modelPath ="E:\\EHES\\WEKAPRUEBAS\\a2model.model";
            emaPath ="E:\\EHES\\WEKAPRUEBAS\\a2emaitzak.txt";
        }else{
            dataPath=args[0];
            trainPath =args[1];
            testPath =args[2];
            blindPath =args[3];
            modelPath =args[4];
            emaPath =args[5];
        }
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataPath);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        Resample r = new Resample();
        r.setRandomSeed(42);
        r.setSampleSizePercent(70);
        r.setNoReplacement(true);
        r.setInvertSelection(false);
        r.setInputFormat(data);
        Instances train = Filter.useFilter(data, r);
        train.setClassIndex(train.numAttributes()-1);

        r.setRandomSeed(42);
        r.setSampleSizePercent(70);
        r.setNoReplacement(true);
        r.setInvertSelection(true);
        r.setInputFormat(data);
        Instances test = Filter.useFilter(data, r);
        test.setClassIndex(test.numAttributes()-1);

        ReplaceWithMissingValue rwmv = new ReplaceWithMissingValue();
        rwmv.setAttributeIndicesArray(new int[]{test.classIndex()});
        rwmv.setIgnoreClass(false);
        rwmv.setProbability(1);
        rwmv.setInputFormat(test);
        Instances blindTest = Filter.useFilter(test, rwmv);
        blindTest.setClassIndex(blindTest.numAttributes()-1);

        //Gorde aurrekoak:
        ConverterUtils.DataSink ds = new ConverterUtils.DataSink(trainPath);
        ds.write(train);
        ds = new ConverterUtils.DataSink(testPath);
        ds.write(test);
        ds = new ConverterUtils.DataSink(blindPath);
        ds.write(blindTest);

        //Klase min
        int minIndex = 0;
        int minMaiz=0;
        for(int i=0; i<data.classAttribute().numValues(); i++){
            int maiz = data.attributeStats(data.classIndex()).nominalCounts[i];
            if(minMaiz==0 && maiz!=0){
                minIndex=i;
                minMaiz=maiz;
            }else if(maiz!=0 && minMaiz>maiz){
                minIndex=i;
                minMaiz=maiz;
            }
        }

        RandomForest rf = new RandomForest();
        double fmax=0.0;
        int trees=1;
        for(int i=1; i<100; i++){
            rf.setNumIterations(i); //No se si es este
            rf.buildClassifier(train);

            Evaluation evaluation = new Evaluation(train);
            evaluation.evaluateModel(rf, test);
            double f = evaluation.fMeasure(minIndex);
            System.out.println(i+" "+f);
            if(f>fmax){
                trees=i;
                fmax=f;
            }
        }
        System.out.println("Fmax: "+fmax);
        System.out.println("Trees: "+trees);
    }
}
