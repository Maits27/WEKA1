package Praktika2;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Randomize;

import java.io.FileNotFoundException;
import java.io.IOException;

public class EntregaSHoldOut1 {
    /*Aurrebaldintzak:
    1. argumentuan: data.arff. Fitxategi horren klasea azken atributuan dator.
    2. argumentuan: train.arff.
    3. argumentuan, dev.arff*/
    public static void main(String[] args) {
        if(args.length==0){
            System.out.println("    Sartutako komandoaren formatua ez da zuzena.\n  Hurrengoko formatua erabili:\n" +
                    "       java -jar HoldOut1.jar /path/to/data.arff /path/to/train.arff /path/to/test.arff");
        }else{
            try {
                String path0, path1, path2 = "";
                path0 = args[0];path1=args[1]; path2=args[2];

                //Datuak dituen fitxategia kargatu:
                ConverterUtils.DataSource source= new ConverterUtils.DataSource(path0);
                Instances data=source.getDataSet();
                data.setClassIndex(data.numAttributes() - 1);

                //Random
                Randomize r = new Randomize();
                r.setRandomSeed(42);
                r.setInputFormat(data);
                Instances rData= Filter.useFilter(data, r);

                StratifiedRemoveFolds sRF = new StratifiedRemoveFolds();
                sRF.setInputFormat(rData);
                sRF.setNumFolds(5);
                Instances test=Filter.useFilter(rData,sRF);

                sRF.setInputFormat(rData);
                sRF.setNumFolds(5);
                sRF.setInvertSelection(true);
                Instances train=Filter.useFilter(rData, sRF);

                ConverterUtils.DataSink.write(path1, train);
                ConverterUtils.DataSink.write(path2, test);
                System.out.println("Guztira "+data.numInstances()+" instantzia egon behar ziren eta guztira "+(train.numInstances()+test.numInstances())+" daude");
                System.out.println("Train instantzia kop: "+train.numInstances());
                System.out.println("Test instantzia kop: "+test.numInstances());
                System.out.println(train.instance(0));
                System.out.println(test.instance(0));

            } catch (FileNotFoundException e) {
                System.out.println("ERROR1: Fitxategi path-a berrikusi:" + args[0]);
            } catch (IOException e) {
                System.out.println("ERROR2: Fitxategi path-aren datuak berrikusi:" + args[0]);
            } catch (Exception e) {e.printStackTrace();}
        }

    }
}
