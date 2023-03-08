package Praktika5;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Instances.*;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Randomize;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

public class Iragarpenak3 {
    /* 2. programan lortutako eredua emanda eta 1. programarekin lortutako test multzoaren iragarpenak egin.
    Kontuz, test multzoak zenbat atributu ditu?

    Test multzoa eredu iragarleak espero dituen atributuetara egokitzeko
    modu ezberdinak daude eta kontzeptualki, ideia garrantzitsuena hauxe da:

    Test multzoan (edo testa bailitzan erabiliko dugun sortan) klasea ezezaguna
    bailitzan jokatu behar dugu, beraz, ezin dugu gainbegiratuak diren metodoak aplikatu.
    Wekan badaude atributuak ezabatzeko filtro gainbegiratu-gabeak ataza hau egiteko.
    Ataza hau egitea, kontzeptualki, garrantzitsua da. */

    public static void main(String[] args) {
        try {
            String testPath, modelPath, dataPath, emaPath;
            if (args.length == 0) {
                System.out.println("Sartutako komando egitura ez da zuzena. Hurrengoko eredua jarraitu:\n" +
                        "java -jar Iragarpen.jar /path/to/data.arff /path/to/test.arff /path/to/karpeta/NB.model");
                dataPath = "E:\\EHES\\WEKAPRUEBAS\\datuenSelekzio.arff";
                testPath = "E:\\EHES\\WEKAPRUEBAS\\70test.arff";
                modelPath = "E:\\EHES\\WEKAPRUEBAS\\nb.model";
                emaPath = "E:\\EHES\\WEKAPRUEBAS\\emaitzakAtributuSelekzio.txt";
            } else {
                dataPath = args[0];
                testPath = args[1];
                modelPath = args[2];
                emaPath = args[3];
            }

            ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            source = new ConverterUtils.DataSource(testPath);
            Instances test = source.getDataSet();
            test.setClassIndex(test.numAttributes() - 1);

            Classifier k = (Classifier) weka.core.SerializationHelper.read(modelPath);

            //TODO
            int atributu[]=new int[data.numAttributes()];
            for(int i=0; i<data.numAttributes(); i++){
                boolean a=false;
                for(int j=0; j<test.numAttributes(); j++){
                    if(!a){
                        if(test.attribute(j).equals(data.attribute(i))){
                            a=true;
                            atributu[i]=j;
                        }
                    }
                }
            }
            //TODO REMOVE CON EL SET INPUT DE TRAIN
            Remove r =new Remove();
            r.setAttributeIndicesArray(atributu);
            r.setInvertSelection(true);
            r.setInputFormat(test);
            Instances rTest = Filter.useFilter(test, r);
            rTest.setClassIndex(rTest.numAttributes()-1);

            FileWriter f = new FileWriter(emaPath);
            BufferedWriter bf = new BufferedWriter(f);
            bf.append("Datuak: "+dataPath);
            bf.append("Test instantziak: "+testPath);
            bf.append("Modeloa: "+modelPath);
            bf.append("\n Klasifikatutako instantziak: \n");
            for(int i=0; i<rTest.numInstances(); i++){
                bf.append(i+" instantzia --->"+k.classifyInstance(rTest.instance(i))+"\n");
            }
            bf.close();

            System.out.println(data.instance(1));
            System.out.println(test.instance(1));
            System.out.println(rTest.instance(1));

        }catch (Exception e){System.out.println(e.toString());}
    }
}
