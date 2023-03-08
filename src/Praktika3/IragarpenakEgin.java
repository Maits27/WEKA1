package Praktika3;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Date;

public class IragarpenakEgin {
    /*Argumentuak:
        1. NB.model: eredua non dagoen esaten digun path (input)
        2. test_blind.arff: instantzien path (input)
        3. test_predictions.txt: iragarpena gordetzeko fitxategiko path (output)*/
    public static void main(String[] args) {
        if(args.length==0){
            System.out.println("    Sartutako komandoaren formatoa ez da zuzena. \n     Hurrengo formatua duen agindua sartu:\n" +
                    "       java -jar Praktika3.IragarpenakEgin.jar /path/to/modeloa.model /path/to/test.arff /path/to/predikzioak.txt");
        }else{
            try {
                String path, path1, path2 = "";
                path = args[0]; path1 = args[1]; path2 = args[2];

                //LOAD MODEL:
                Classifier klasifikadore = (Classifier) weka.core.SerializationHelper.read(path);

                ConverterUtils.DataSource source= new ConverterUtils.DataSource(path1);
                Instances testData=source.getDataSet();
                testData.setClassIndex(testData.numAttributes() - 1);

                //Kalitatearen estimazioa eman:
                FileWriter file=new FileWriter(path2);
                BufferedWriter bf=new BufferedWriter(file);

                bf.append(new Date().toString());
                bf.newLine(); bf.newLine();
                bf.append("Test datuen path-a: "+path1);
                bf.newLine(); bf.newLine();
                bf.append("Test instantzia kopurua: "+testData.numInstances());
                bf.newLine(); bf.newLine();
                //bf.append("Klaseak har ditzaken balioak: ");
                bf.append("KLASIFIKAZIOA:");
                bf.newLine();
                for(int i=0; i<testData.numInstances(); i++){
                    double iragarpena= klasifikadore.classifyInstance(testData.instance(i));
                    bf.append(i+ " instantziaren klasea: "+ testData.classAttribute().value((int)iragarpena));
                    bf.newLine();
                }
                bf.close();
            }catch (Exception e){System.out.println(e.toString());}
        }
    }
}
