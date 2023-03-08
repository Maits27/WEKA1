package Praktika2;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Randomize;

import java.io.FileNotFoundException;
import java.io.IOException;

public class StratifiedHoldOut1 {
    /*Aurrebaldintzak:
    1. argumentuan: train.arff. Fitxategi horren klasea azken atributuan dator.
    2. argumentuan: dev.arff.
    3. argumentuan, evaluation.txt, irteerarako emaitzak gordetzeko fitxategi baten path-a ematen da*/
    public static void main(String[] args) {
        try {

            String path0, path1, path2 = "";
            System.out.println(System.getProperty("user.dir") + "\n");

            if (args.length==0) {
                path0 = "C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\2LAB\\2. Praktika Datuak-20230129\\adult.train.arff";
                path1 = "C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\2LAB\\2. Praktika Datuak-20230129\\strain2.arff";
                path2 = "C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\2LAB\\2. Praktika Datuak-20230129\\stest2.arff";
            } else {path0 = args[0];path1=args[1]; path2=args[2];}

            //Datuak dituen fitxategia kargatu:
            ConverterUtils.DataSource source= new ConverterUtils.DataSource(path0);
            Instances data=source.getDataSet();

            //Kasu honetan aurrebaldintzatan jartzen duen moduan klasea azkenengo atributua da

            if (data.classIndex() == -1) {data.setClassIndex(data.numAttributes() - 1);}
            Randomize r = new Randomize();
            r.setRandomSeed(42);
            r.setInputFormat(data);
            Instances rData= Filter.useFilter(data, r);

            Resample resample=new Resample();
            resample.setInputFormat(rData);
            resample.setSampleSizePercent(80);
            resample.setInvertSelection(true);
            resample.setNoReplacement(true);
            Instances test= Filter.useFilter(rData, resample);

            resample.setInputFormat(rData);
            resample.setNoReplacement(true);
            resample.setSampleSizePercent(80);
            Instances train=Filter.useFilter(rData, resample);

            System.out.println("Guztira "+data.numInstances()+" instantzia egon behar ziren eta guztira "+(train.numInstances()+test.numInstances())+" daude");
            System.out.println("Train instantzia kop: "+train.numInstances());
            System.out.println("Test instantzia kop: "+test.numInstances());
            System.out.println(train.instance(0));
            System.out.println(test.instance(0));

            ConverterUtils.DataSink.write(path1, train);
            ConverterUtils.DataSink.write(path2, test);

        } catch (FileNotFoundException e) {
            System.out.println("ERROR1: Fitxategi path-a berrikusi:" + args[0]);
        } catch (IOException e) {
            System.out.println("ERROR2: Fitxategi path-aren datuak berrikusi:" + args[0]);
        } catch (Exception e) {e.printStackTrace();}
    }
}
