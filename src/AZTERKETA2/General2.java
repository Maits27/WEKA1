package AZTERKETA2;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;
import weka.filters.unsupervised.instance.Resample;

import java.util.Random;

import static weka.classifiers.lazy.IBk.*;

public class General2 {

    public static void main(String[] args) throws Exception {
        String dataPath;
        if(args.length==0){
            dataPath="E:\\EHES\\WEKAPRUEBAS\\neutrons.arff";
        }else{
            dataPath=args[0];
        }
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataPath);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        Resample r = new Resample();
        r.setRandomSeed(42);
        r.setInvertSelection(false);
        r.setNoReplacement(true);
        r.setSampleSizePercent(70);
        r.setInputFormat(data);
        Instances train = Filter.useFilter(data, r);
        train.setClassIndex(train.numAttributes()-1);

        r.setRandomSeed(42);
        r.setInvertSelection(true);
        r.setNoReplacement(true);
        r.setSampleSizePercent(70);
        r.setInputFormat(data);
        Instances test = Filter.useFilter(data, r);
        test.setClassIndex(test.numAttributes()-1);

        System.out.println(data.numInstances()+" "+data.classIndex());
        System.out.println(train.numInstances()+" "+train.classIndex());
        System.out.println(test.numInstances()+" "+test.classIndex());



        LinearNNSearch euclidean = new LinearNNSearch();
        euclidean.setDistanceFunction(new EuclideanDistance());
        LinearNNSearch manhattan = new LinearNNSearch();
        manhattan.setDistanceFunction(new ManhattanDistance());
        LinearNNSearch minkowski = new LinearNNSearch();
        minkowski.setDistanceFunction(new MinkowskiDistance());
        LinearNNSearch filtered = new LinearNNSearch();
        filtered.setDistanceFunction(new FilteredDistance());
        LinearNNSearch chev = new LinearNNSearch();
        chev.setDistanceFunction(new ChebyshevDistance());

        LinearNNSearch dist[] = new LinearNNSearch[]{euclidean, minkowski, manhattan, chev, filtered};

        SelectedTag weights[] = new SelectedTag[]{new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING),
                new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING),
                new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING)};

        int kaux=0;
        SelectedTag waux=null;
        LinearNNSearch daux =null;
        double fmax = 0.0;
        IBk ibk = new IBk();

        for (int k = 1; k<(data.numInstances()/4); k++){ //Auzokide kop laurdenerarte
            //ibk.setKNN(k);
            for (LinearNNSearch d:dist){
                //ibk.setNearestNeighbourSearchAlgorithm(d);
                for (SelectedTag w:weights){
                    ibk = new IBk();
                    ibk.setKNN(k);
                    ibk.setNearestNeighbourSearchAlgorithm(d);
                    ibk.setDistanceWeighting(w);
                    ibk.buildClassifier(train);

                    Evaluation evaluation = new Evaluation(train);
                    evaluation.evaluateModel(ibk, test);

                    double f = evaluation.weightedFMeasure();
                    if(f>fmax){
                        fmax=f;
                        kaux=k;
                        daux=d;
                        waux=w;
                        System.out.println("Fmeasure hobeagoa lortu da: "+fmax);
                    }
                }
            }
        }
        System.out.println("Lortu egin diren parametroak:");
        System.out.println("Distantzia: "+ daux.getDistanceFunction().getClass());
        System.out.println("Weight mode: "+ waux);
        System.out.println("K balioa: "+ kaux);

        ibk= new IBk();
        ibk.setKNN(kaux);
        ibk.setNearestNeighbourSearchAlgorithm(daux);
        ibk.setDistanceWeighting(waux);
        ibk.buildClassifier(train);

        Evaluation evaluation = new Evaluation(train);
        evaluation.evaluateModel(ibk, test);

        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toClassDetailsString());
        System.out.println(evaluation.toMatrixString());

        ReplaceWithMissingValue replace = new ReplaceWithMissingValue();
        replace.setProbability(1);
        replace.setIgnoreClass(false);
        replace.setAttributeIndicesArray(new int[]{test.classIndex()});
        replace.setInputFormat(test);
        Instances blind = Filter.useFilter(test, replace);
        blind.setClassIndex(blind.numAttributes()-1);

        for (int i=0; i<blind.numInstances(); i++){
            System.out.println(i+" instantzia ---> "+ ibk.classifyInstance(blind.instance(i)));
        }
    }

}
