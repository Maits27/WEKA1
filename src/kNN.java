import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import javax.swing.*;
import javax.swing.plaf.basic.BasicInternalFrameTitlePane;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Date;
import java.util.Random;

//import weka.classifiers.meta.GridSearch;
public class kNN {
    /*Parametroen artean, besteak beste, honako hauek daude:
        k: auzokide kopurua (KNN);
        d: metrika (nearestNeighbourSearchAlgorithm → distanceFunction );
        w: distantziaren ponderazio faktorea (distanceWeighting).
      Helburua:
        k parametroaren balioak ekortu eta 10-fold cross validation eginda f-measure
        neurria maximizatzen duen parametroa itzuli
    */
    //https://waikato.github.io/weka-wiki/optimizing_parameters/
    //https://java.hotexamples.com/examples/weka.classifiers.meta/CVParameterSelection/-/java-cvparameterselection-class-examples.html
    public static void main(String[] args) {
        try {
            String path, path1, path2 = "";
            System.out.println(System.getProperty("user.dir") + "\n");

            if (args.length==0) {
                path = "C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\3LAB\\3. Praktika Datuak-20230207\\data_1\\devr.arff";
                path1 ="C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\3LAB\\3. Praktika Datuak-20230207\\NB.model";
                path2 ="C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\3LAB\\3. Praktika Datuak-20230207\\Praktika3.EreduaSortu.txt";
            } else {
                path = args[0];
                path1 = args[1];
                path2 = args[2];
            }

            ConverterUtils.DataSource source= new ConverterUtils.DataSource(path);
            Instances data=source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);



        }catch (Exception e){
            System.out.println(e.toString());
        }
    }
}
