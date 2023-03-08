package AZTERKETA2;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AttributeSelectedClassifier;
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

public class General {
    /*
    * Edukia: inplementazioa Javan Wekako liburutegiak integratuz. Kurtso hasieratik une honetara arte jorratutakoa sartzen da. Besteak-beste:
        Datuak jaso/gorde
        Datuen analisia
        Eredu optimoa lortu parametro ekorketa eginez
        Eredua gorde/kargatu
        Test multzo bateko instantzien iragarritako klasea eman
        Itxarondako kalitatea lortzeko ebaluazio-teknikak: hold-out (run anitzen avg eta stdev), k-fold cross validation, ez-zintzoa
        Ebaluazio eskemak
        Ebaluazio metrikak klaseka eta batazbestekoak
        Atributuen hautaketa (train vs test)
        Filtroak
        Runable jar*/
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

        //AUKERATU KNN PARAMETROAK:
        IBk ibk = new IBk();
        ibk.buildClassifier(train);

        LinearNNSearch mikowski = new LinearNNSearch();
        mikowski.setDistanceFunction(new MinkowskiDistance());
        LinearNNSearch euclidean = new LinearNNSearch();
        euclidean.setDistanceFunction(new EuclideanDistance());
        LinearNNSearch manhattan = new LinearNNSearch();
        manhattan.setDistanceFunction(new ManhattanDistance());
        LinearNNSearch filtered = new LinearNNSearch();
        filtered.setDistanceFunction(new FilteredDistance());
        LinearNNSearch chev = new LinearNNSearch();
        chev.setDistanceFunction(new ChebyshevDistance());
        LinearNNSearch lnns_zerrenda[] = new LinearNNSearch[]{euclidean, manhattan, mikowski, filtered, chev};

        SelectedTag[] st_zerrenda = new SelectedTag[]{new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING), new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING),
                new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING)};

        SelectedTag st=null;
        LinearNNSearch lnns= null;
        int k=0;
        double fmax=0;

        for (int i = 1; i<(train.numInstances()/4); i++){ //Auzokide kop laurdenerarte
            ibk.setKNN(i);
            for (LinearNNSearch d:lnns_zerrenda){
                ibk.setNearestNeighbourSearchAlgorithm(d);
                for (SelectedTag w:st_zerrenda){
                    ibk.setDistanceWeighting(w);
                    ibk.buildClassifier(train);

                    // Ebaluatzailea sortu:
                    Evaluation eval = new Evaluation(train);
                    eval.evaluateModel(ibk, test);

                    double f = eval.weightedFMeasure();

                    if (f > fmax){
                        fmax = f;
                        k = i;
                        lnns = d;
                        st = w;
                        System.out.println("Lortu da hobekuntza, oraingo F-Measure: "+fmax);
                    }
                }
            }
        }

        FileWriter f = new FileWriter(emaPath);
        BufferedWriter bf = new BufferedWriter(f);
        bf.append("ERABILIKO DIREN KLASIFIKADORE PARAMETROAK: \n" +
                "K PARAMETROA: " +k+"\n"+
                "DISTANTZIA PARAMETROA: " +lnns.getDistanceFunction().getClass()+"\n"+
                "WEIGHT PARAMETROA: "+st+"\n");
        System.out.println("DISTANTZIA PARAMETROA: " +lnns.getDistanceFunction().getClass());

        //GORDE KLASIFIKADOREA:
        ibk.setKNN(k);
        ibk.setDistanceWeighting(st);
        ibk.setNearestNeighbourSearchAlgorithm(lnns);
        ibk.buildClassifier(data);
        weka.core.SerializationHelper.write(modelPath, ibk);

        //Atribute selection classifier:
        ibk.setKNN(k);
        ibk.setDistanceWeighting(st);
        ibk.setNearestNeighbourSearchAlgorithm(lnns);
        ibk.buildClassifier(train);

        AttributeSelectedClassifier asc = new AttributeSelectedClassifier();
        asc.setSearch(new BestFirst());
        asc.setEvaluator(new CfsSubsetEval());
        asc.setClassifier(new NaiveBayes());
        asc.setClassifier(ibk);

        //EBALUAZIOA:
        Evaluation evaluation = new Evaluation(train);
        evaluation.evaluateModel(asc, test);

        bf.append("\n EBALUAKETA PARAMETROEN SELEKZIOAREKIN: \n");
        bf.append(evaluation.toSummaryString());
        bf.append(evaluation.toClassDetailsString());
        bf.append(evaluation.toMatrixString());

        //KLASIFIKAZIOA:
        bf.append("\n INSTANTZIEN KLASIFIKAZIOA SELEKZIOAREKIN: \n");
        for (int i =0; i<blindTest.numInstances(); i++){
            bf.append(i+" instantzia ---> "+ibk.classifyInstance(blindTest.instance(i))+"\n");
        }

        bf.close();

    }
}
