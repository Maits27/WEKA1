package Praktika5;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceWithMissingValue;
import weka.filters.unsupervised.instance.Resample;

public class Multzoak1 {
    /* ARGUMENTUAK:
        1. data.arff: jatorrizko datuen path (input)
        2. train.arff datuak gordetzeko path (output)
        3. test_blind.arff datuak gordetzeko path (output)*/
    public static void main(String[] args) {
        try {
            String dataPath, trainPath, testPath;
            if(args.length==0){
                System.out.println("Sartutako komando egitura ez da zuzena. Hurrengoko eredua jarraitu:\n" +
                        "java -jar Stratified70percentSplit.jar  /path/to/data.arff  /path/to/irteerako/train.arff " +
                        " /path/to/irteerako/karpeta/test_blind.arff");
                dataPath= "E:\\EHES\\WEKAPRUEBAS\\adult.train.arff";
                trainPath= "E:\\EHES\\WEKAPRUEBAS\\70train.arff";
                testPath= "E:\\EHES\\WEKAPRUEBAS\\70test.arff";
            }else{
                dataPath=args[0];
                trainPath=args[1];
                testPath=args[2];
            }
            ConverterUtils.DataSource source= new ConverterUtils.DataSource(dataPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);

            Resample resample = new Resample();
            resample.setRandomSeed(42);
            resample.setNoReplacement(true);
            resample.setSampleSizePercent(70);
            resample.setInvertSelection(false);
            resample.setInputFormat(data);

            ConverterUtils.DataSink dataSink= new ConverterUtils.DataSink(trainPath);
            dataSink.write(Filter.useFilter(data,resample));

            resample.setRandomSeed(42);
            resample.setNoReplacement(true);
            resample.setSampleSizePercent(70);
            resample.setInvertSelection(true);
            resample.setInputFormat(data);

            Instances test = Filter.useFilter(data,resample);
            test.setClassIndex(test.numAttributes()-1);

            //TODO ReplaceWithMissingValue ???
            //for (int j =0; j<test.numInstances();j++){test.instance(j).setClassMissing();}
            ReplaceWithMissingValue rpwmv = new ReplaceWithMissingValue();
            rpwmv.setIgnoreClass(true);
            rpwmv.setProbability(1);
            rpwmv.setAttributeIndicesArray(new int[]{test.classIndex()});
            rpwmv.setInputFormat(test);
            test=Filter.useFilter(test, rpwmv);

            for (int j =0; j<test.numInstances();j++){System.out.println(test.instance(0));}
            dataSink= new ConverterUtils.DataSink(testPath);
            dataSink.write(test);


        }catch (Exception e){System.out.println(e.toString());}
    }
}
