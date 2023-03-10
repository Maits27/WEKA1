package AZTERKETA2;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.FileWriter;
import java.util.Random;

public class EbazpenaAingeru {
    public static void main(String[] args) {
        try {
            // 0. Datuak kargatu
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            System.out.println("Data multzoak " + data.numInstances() + " instantzia ditu.");

            ConverterUtils.DataSource test_source = new ConverterUtils.DataSource(args[2]);
            Instances blind_test = test_source.getDataSet();
            blind_test.setClassIndex(blind_test.numAttributes() - 1);

            // 1. Sailkatzaile optimoa
            int klase_min_index = Utils.minIndex(data.attributeStats(data.classIndex()).nominalCounts);

            // 1.1. Estratifikazioa
            Resample filter_resample = new Resample();
            filter_resample.setRandomSeed(1); // Randomize barrutik egin dezan
            filter_resample.setNoReplacement(true);
            filter_resample.setSampleSizePercent(70.0);
            filter_resample.setInputFormat(data);
            Instances train_data = Filter.useFilter(data, filter_resample);
            train_data.setClassIndex(train_data.numAttributes() - 1);

            filter_resample.setInvertSelection(true);
            filter_resample.setInputFormat(data);
            Instances test_data = Filter.useFilter(data, filter_resample);
            test_data.setClassIndex(test_data.numAttributes() - 1);

            System.out.println("Train multzoaren instantzia kopurua: " + train_data.numInstances());
            System.out.println("Test multzoaren instantzia kopurua: " + test_data.numInstances());

            // 1.2. Parametro ekorketa
            SMO smo_sailkatzaile = new SMO();
            PolyKernel polyKernel = new PolyKernel();

            double max_exp = 0;
            double max_f_meas = 0;
            for (double exp = 1; exp <= 4; exp = exp + 0.1) {
                polyKernel.setExponent(exp);
                smo_sailkatzaile.setKernel(polyKernel);
                smo_sailkatzaile.buildClassifier(train_data);

                Evaluation evaluation = new Evaluation(train_data);
                evaluation.evaluateModel(smo_sailkatzaile, test_data);

                double f_meas = evaluation.fMeasure(klase_min_index);
                System.out.println(exp + " exponenetearekin, " + f_meas + " balioa lortu da.");
                if (f_meas > max_f_meas) {
                    max_f_meas = f_meas;
                    max_exp = exp;
                }
            }
            System.out.println("SMO optimoa lortzeko behar den exponenentea " + max_exp + " da.");

            // 1.3. Sailkatzailea gorde
            SMO sailkatzaile_optimo = new SMO();
            polyKernel.setExponent(max_exp);
            sailkatzaile_optimo.setKernel(polyKernel);

            sailkatzaile_optimo.buildClassifier(data); // datu guztiekin build egin
            SerializationHelper.write(args[1], sailkatzaile_optimo);


            // 2. Sailkatzailearen kalitatearen estimazioa
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(sailkatzaile_optimo, data, 4, new Random(1));

            System.out.println("Klase minoriatarioaren f-measure: " + evaluation.fMeasure(klase_min_index));
            System.out.println(evaluation.toMatrixString("Evaluazio matrizea"));


            // 3. Test multzoko iragarpenak
            SMO sailkatzaile_iragarle = (SMO) SerializationHelper.read(args[1]);
            FileWriter writer = new FileWriter(args[3]);
            for (int i = 0; i < blind_test.numInstances(); i++) {
                double pred = sailkatzaile_iragarle.classifyInstance(blind_test.instance(i));
                writer.write(i + " --> " + blind_test.classAttribute().value((int) pred) + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}