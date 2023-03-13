package AZTERKETA2;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

public class ebazpena {
    public static void main(String[] args) throws Exception {
        String dataPath, modelPath, blindPath, predictionsPath;
        if(args.length==0){
            System.out.println("Sartu formatu hau: java -jar /path/to/data.arff" +
                    "/path/to/smo.model /path/to/test_blind.arff /path/to/test_blind_predictions.txt");
            args = new String[]{"data_supervised.arff", "model.model", "data_test_blind.arff", "test_blind_predictions.txt"};
        }else{
            dataPath=args[0];
            modelPath =args[1];
            blindPath =args[2];
            predictionsPath =args[3];
        }
        dataPath=args[0];
        modelPath =args[1];
        blindPath =args[2];
        predictionsPath =args[3];

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataPath);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        int minIndex=0;
        int minMaiz=0;
        for (int i =0; i<data.classAttribute().numValues(); i++){
            int m = data.attributeStats(data.classIndex()).nominalCounts[i];
            if(minMaiz==0 && m!=0){
                minMaiz=m;
                minIndex=i;
            }else if(m!=0 && m<minMaiz){
                minMaiz=m;
                minIndex=i;
            }
        }

        Resample r = new Resample();
        r.setInvertSelection(false);
        r.setNoReplacement(true);
        r.setRandomSeed(42);
        r.setSampleSizePercent(70);
        r.setInputFormat(data);
        Instances train = Filter.useFilter(data, r);
        train.setClassIndex(train.numAttributes()-1);

        r.setInvertSelection(true);
        r.setNoReplacement(true);
        r.setRandomSeed(42);
        r.setSampleSizePercent(70);
        r.setInputFormat(data);
        Instances test = Filter.useFilter(data, r);
        test.setClassIndex(test.numAttributes()-1);

        SMO smo = new SMO();
        PolyKernel pk = new PolyKernel();

        double exMax=1;
        double fmax=0.0;
        for (double i=1; i<5; i++){
            pk.setExponent(i);
            smo.setKernel(pk);
            smo.buildClassifier(train);

            Evaluation evaluation= new Evaluation(train);
            evaluation.evaluateModel(smo, test);

            double f = evaluation.fMeasure(minIndex);
            System.out.println(i+" exponentearekin ateratako fmeasure: "+f);
            if(f>fmax){
                fmax=f;
                exMax=i;
            }
        }

        //eredua gorde:
        pk.setExponent(exMax);
        smo.setKernel(pk);
        smo.buildClassifier(data);
        weka.core.SerializationHelper.write(modelPath, smo);


        //1.2
        pk.setExponent(exMax);
        smo.setKernel(pk);
        //smo.buildClassifier(train); //TODO EZ DA BUILD EGIN BEHAR

        Evaluation evaluation= new Evaluation(train);
        evaluation.crossValidateModel(smo, train, 4, new Random());

        System.out.println("Klase minoritarioaren fmeasure: "+ evaluation.fMeasure(minIndex));
        System.out.println("Nahasmen matrize: \n"+ evaluation.toMatrixString());

        //1.3
        Classifier k = (Classifier) weka.core.SerializationHelper.read(modelPath);
        source= new ConverterUtils.DataSource(blindPath);
        Instances blind=source.getDataSet();
        blind.setClassIndex(blind.numAttributes()-1);

        FileWriter f = new FileWriter(predictionsPath);
        BufferedWriter bf = new BufferedWriter(f);

        bf.append("IRAGARPENAK: \n");

        for(int i=0; i< blind.numInstances(); i++){
            double z=k.classifyInstance(blind.instance(i));
            bf.append((i+1) +" instantzia ---> "+blind.classAttribute().value((int) z)+"\n");
        }
        bf.close();
    }
}