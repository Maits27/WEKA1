package AZTERKETA2;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.Resample;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import static weka.classifiers.lazy.IBk.*;

public class UnPoquitoDeTo {
    public static void main(String[] args) throws Exception{
        String dataPath, testPath, trainPath, blindPath, modelPath, emaPath;
        if(args.length==0){
            dataPath = "E:\\EHES\\WEKAPRUEBAS\\neutrons.arff";
            trainPath = "E:\\EHES\\WEKAPRUEBAS\\trainAzterketaProba.arff";
            testPath = "E:\\EHES\\WEKAPRUEBAS\\testAzterketaProba.arff";
            blindPath = "E:\\EHES\\WEKAPRUEBAS\\blindAzterketaProba.arff";
            modelPath = "E:\\EHES\\WEKAPRUEBAS\\modelAzterketaProba.model";
            emaPath = "E:\\EHES\\WEKAPRUEBAS\\emaAzterketaProba.txt";
        }else{
            dataPath = args[0];
            trainPath = args[1];
            testPath = args[2];
            blindPath = args[3];
            modelPath = args[4];
            emaPath = args[5];
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

        //BLIND TEST-A SORTU:
        ReplaceWithMissingValue rwmv = new ReplaceWithMissingValue();
        rwmv.setIgnoreClass(false);
        rwmv.setProbability(1);
        rwmv.setAttributeIndicesArray(new int[]{test.classIndex()});
        rwmv.setInputFormat(test);
        Instances blind = Filter.useFilter(test, rwmv);
        blind.setClassIndex(blind.numAttributes()-1);

        //Klasifikadorea ibk
        IBk ibk= new IBk();
        ibk.buildClassifier(train);

        LinearNNSearch euc = new LinearNNSearch();
        euc.setDistanceFunction(new EuclideanDistance());
        LinearNNSearch manh = new LinearNNSearch();
        manh.setDistanceFunction(new ManhattanDistance());
        LinearNNSearch mink = new LinearNNSearch();
        mink.setDistanceFunction(new MinkowskiDistance());
        LinearNNSearch filt = new LinearNNSearch();
        filt.setDistanceFunction(new FilteredDistance());
        LinearNNSearch chev = new LinearNNSearch();
        chev.setDistanceFunction(new ChebyshevDistance());

        LinearNNSearch dist[] = new LinearNNSearch[]{euc, mink, manh, filt, chev};

        SelectedTag weights[] = new SelectedTag[]{new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING),
        new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING), new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING)};

        int kaux=0;
        LinearNNSearch daux=null;
        SelectedTag waux=null;
        double fmax=0.0;
        int it=0;

        for(int k=1; k<train.numInstances()/4; k++){
            for(LinearNNSearch d: dist){
                for (SelectedTag w: weights){
                    ibk.setKNN(k);
                    ibk.setNearestNeighbourSearchAlgorithm(d);
                    ibk.setDistanceWeighting(w);
                    ibk.buildClassifier(train);

                    Evaluation evaluation = new Evaluation(train);
                    evaluation.crossValidateModel(ibk, test, 10, new Random(1));

                    double f = evaluation.weightedFMeasure();
                    System.out.println(it++);
                    if(f>fmax){
                        System.out.println("Lortu da fmeasure maximo berria: "+f);
                        fmax=f;
                        waux=w;
                        daux=d;
                        kaux=k;
                    }
                }
            }
        }
        ibk.setKNN(kaux);
        ibk.setDistanceWeighting(waux);
        ibk.setNearestNeighbourSearchAlgorithm(daux);

        AttributeSelectedClassifier asc = new AttributeSelectedClassifier();
        asc.setEvaluator(new CfsSubsetEval());
        asc.setSearch(new BestFirst());
        asc.setClassifier(ibk);
        asc.buildClassifier(data);

        //Gorde modeloa:
        weka.core.SerializationHelper.write(modelPath, asc);
        Classifier k = (Classifier) weka.core.SerializationHelper.read(modelPath);

        FileWriter f = new FileWriter(emaPath);
        BufferedWriter bf = new BufferedWriter(f);

        int minMaiz=0;
        int minIndex=0;
        for(int i =0; i<data.classAttribute().numValues(); i++){
            int m = data.attributeStats(data.classIndex()).nominalCounts[i];
            if(minMaiz==0 && m!=0){
                minMaiz=m;
                minIndex=i;
            }else if(m!=0 && minMaiz>m){
                minMaiz=m;
                minIndex=i;
            }
        }

        bf.append("DATUEI BURUZKO INFORMAZIOA:" +
                "\n     Instantzia kopurua: " + data.numInstances()+
                "\n     Atributu kopurua: " + data.numAttributes()+
                "\n     "+data.attribute(0)+" atributuaren missing balio kopurua: " + data.attributeStats(0).missingCount+
                "\n     "+data.attribute(0)+" atributuaren unique balio kopurua: " + data.attributeStats(0).uniqueCount+
                "\n     Klase balio ezberdin kopurua: " + data.numDistinctValues(data.classIndex())+
                "\n     KLASE MINORITARIOA: " + data.classAttribute().value(minIndex)+
                "\n     KLASE MINORITARIOAREN MAIZTASUNA: "+minMaiz);
        bf.append("\n AUKERATUTAKO PARAMETROAK IBK KLASIFIKADOREARENTZAT: " +
                "\n     K: " + kaux+
                "\n     DISTANCE: " + daux.getDistanceFunction().getClass()+
                "\n     WEIGHT: " +waux);

        Evaluation evaluation2= new Evaluation(train);
        evaluation2.evaluateModel(asc, test);
        bf.append("\n EBALUAZIOA PARAMETRO EKORKETAREKIN:" +
                "\n     Accuracy: " + evaluation2.pctCorrect()+
                "\n     Klase minoritarioaren presizioa: " + evaluation2.precision(minIndex)+
                "\n     Klase minoritarioaren recall: " + evaluation2.recall(minIndex)+
                "\n     Klase minoritarioaren fmeasure: " + evaluation2.fMeasure(minIndex)+
                "\n     " + evaluation2.toSummaryString()+
                "\n     " +evaluation2.toClassDetailsString()+
                "\n     "+evaluation2.toMatrixString());

        bf.append("\n KLASIFIKATZEKO INSTANTZIAK: ");
        for(int i =0; i<blind.numInstances(); i++){
            bf.append("\n"+i+" instantzia ---> "+k.classifyInstance(blind.instance(i)));
        }

        bf.close();

        //Gorde datuak ARFF fitxategietan:
        ConverterUtils.DataSink ds = new ConverterUtils.DataSink(trainPath);
        ds.write(train);
        ds = new ConverterUtils.DataSink(testPath);
        ds.write(test);
        ds = new ConverterUtils.DataSink(blindPath);
        ds.write(blind);
    }
}
