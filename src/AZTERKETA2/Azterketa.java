package AZTERKETA2;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.LinearNNSearch;

import java.io.FileWriter;
import java.util.Random;

public class Azterketa {
    public static void main(String[] args) {
        try {
            String dataPath, blindPath, modelPath, evalPath, predictionsPath;
            if(args.length==0){
                System.out.println("Ez da formato zuzena sartu.");
                dataPath="E:\\EHES\\WEKAPRUEBAS\\data_supervised.arff";
                blindPath ="E:\\EHES\\WEKAPRUEBAS\\data_test_blind.arff";
                modelPath ="E:\\EHES\\WEKAPRUEBAS\\smo.model";
                evalPath ="E:\\EHES\\WEKAPRUEBAS\\SMOeval.txt";
                predictionsPath ="E:\\EHES\\WEKAPRUEBAS\\azt2_test_blind_predictions.txt";
            }else{
                dataPath=args[0];
                blindPath =args[1];
                modelPath =args[2];
                evalPath =args[3];
                predictionsPath =args[4];
            }

            ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);

            //Datuen analisia:
            System.out.println("ATRIBUTU NOMINALEN DISTINCT VALUES:");
            for(int i = 0; i<data.numAttributes(); i++){
                if(data.attribute(i).isNominal()){
                    System.out.println(i+". "+data.attribute(i).name()+" atributuak dituen balio ezberdin kop: "+data.attributeStats(i).distinctCount);
                }
            }
            System.out.println("\nKLASE BALIOEN MAIZTASUNAK:");
            for(int i = 0; i<data.classAttribute().numValues(); i++){
                System.out.println(data.classAttribute().value(i)+" balioaren maiztasuna: "+data.attributeStats(data.classIndex()).nominalCounts[i]);
            }
            //int minIndex=Utils.minIndex(data.attributeStats(data.classIndex()).nominalCounts);
            int minIndex=0;
            int minCount=0;
            for(int i = 0; i<data.classAttribute().numValues(); i++){
                int c = data.attributeStats(data.classIndex()).nominalCounts[i];
                if(minCount==0 && c!=0){
                    minIndex=i;
                    minCount=c;
                }else if(c<minCount && c!=0){
                    minIndex=i;
                    minCount=c;
                }
            }
            System.out.println("KLASE MINORITARIOA: "+ data.classAttribute().value(minIndex)+"\n" +
                    "ETA BERE MAIZTASUNA: "+data.attributeStats(data.classIndex()).nominalCounts[minIndex]);

            //Sailkatzaile optimoa:
            System.out.println("\nEXPONENTE MAXIMOA LORTZEKO:\n");
            Evaluation eval = new Evaluation(data);
            weka.classifiers.functions.SMO smo = new SMO();
            double fmax =0;
            int expmax=0;
            weka.classifiers.functions.supportVector.PolyKernel pk = new PolyKernel();
            System.out.println("Exponente maximoa aldatu da, orain da: "+expmax);
            for(int i = 0; i<6; i++){
                pk.setExponent(i);
                smo.setKernel(pk);
                smo.buildClassifier(data);
                eval.crossValidateModel(smo, data, 3, new Random(1));
                System.out.println(i+" exponentearekin lortutako f-measure: "+eval.weightedFMeasure());
                if(eval.weightedFMeasure()>fmax){
                    expmax=i;
                    fmax=eval.weightedFMeasure();
                    System.out.println("Exponente maximoa aldatu da, orain da: "+expmax);
                }
            }
            pk.setExponent(expmax);
            smo.setKernel(pk);
            smo.buildClassifier(data);

            weka.core.SerializationHelper.write(modelPath, smo);

            //Sailkatzaile kalitate estimazioa:
            FileWriter f = new FileWriter(evalPath);
            f.append("EBALUAZIO EZ ZINTZOAREN EMAITZAK: \n");
            eval.evaluateModel(smo, data);
            f.append("KLASE MINORITARIOAREKIKO: \n" +
                    "PRECISION:" +eval.precision(minIndex)+"\n"+
                    "RECALL: " +eval.recall(minIndex)+"\n"+
                    "F-MEASURE: "+eval.fMeasure(minIndex)+"\n");
            f.append("NAHASMEN MATRIZEA: \n"+eval.toMatrixString());
            f.close();

            //Test iragarpenak
            source = new ConverterUtils.DataSource(blindPath);
            Instances test = source.getDataSet();
            test.setClassIndex(test.numAttributes()-1);

            Classifier k = (Classifier) weka.core.SerializationHelper.read(modelPath);

            f = new FileWriter(predictionsPath);
            f.append("PREDIKZIOEN EMAITZAK: \n");
            for(int i = 0; i<test.numInstances(); i++){
                f.append(i+" instantzia ----> "+k.classifyInstance(test.instance(i))+"\n");
            }
            f.close();
        }catch (Exception e){System.out.println(e.toString());}
    }
}
