//package dataminingexample;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
//import weka.classifiers.lazy.IB1;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;


///////////////////////////////////////////////////////
// Observa: http://weka.wikispaces.com/Use+Weka+in+your+Java+code
///////////////////////////////////////////////////////
public class DataMiningExample {

    public static void main(String[] args) throws Exception {
        /////////////////////////////////////////////////////////////
        FileReader fi=null;
        try {
            fi= new FileReader("C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\2LAB\\2. Praktika Datuak-20230129\\adult.train.arff");
        } catch (FileNotFoundException e) {
            System.out.println("ERROR: Revisar path del fichero de datos:"+args[0]);
        }

        Instances data=null;
        try {
            data = new Instances(fi);
        } catch (IOException e) {
            System.out.println("ERROR: Revisar contenido del fichero de datos: "+args[0]);
        }

        try {
            fi.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
        }

        data.setClassIndex(data.numAttributes()-1);


        /////////////////////////////////////////////////////////////
        AttributeSelection filter= new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search=new BestFirst();
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);

        Instances newData = Filter.useFilter(data, filter);



        /////////////////////////////////////////////////////////////
        //
        NaiveBayes estimador= new NaiveBayes();

        //
        Evaluation evaluator = new Evaluation(newData);
        evaluator.crossValidateModel(estimador, newData, 10, new Random(1)); // Random(1): the seed=1 means "no shuffle" :-!
        //
        double acc=evaluator.pctCorrect();
        double inc=evaluator.pctIncorrect();
        double kappa=evaluator.kappa();
        double mae=evaluator.meanAbsoluteError();
        double rmse=evaluator.rootMeanSquaredError();
        double rae=evaluator.relativeAbsoluteError();
        double rrse=evaluator.rootRelativeSquaredError();
        double confMatrix[][]= evaluator.confusionMatrix();

        System.out.println("Correctly Classified Instances  " + acc);
        System.out.println("Incorrectly Classified Instances  " + inc);
        System.out.println("Kappa statistic  " + kappa);
        System.out.println("Mean absolute error  " + mae);
        System.out.println("Root mean squared error  " + rmse);
        System.out.println("Relative absolute error  " + rae);
        System.out.println("Root relative squared error  " + rrse);

    }
}