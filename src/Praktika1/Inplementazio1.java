package Praktika1;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Enumeration;

public class Inplementazio1 {
    public static void main(String[] args) {
        try{
            /*Aurrebaldintzak:
                1. argumentuan .arff fitxategi baten path-a hartzen da.
                Fitxategi horren klasea azken atributuan dator. */

            /*
            //Eskaner bidez eskatu path hori:
            System.out.println("Sartu fitxategiaren path osoa: ");
            Scanner sc=new Scanner(System.in);
            String path=sc.nextLine();
            */

            //Path-a inportatzeko artxiboraino zuzenean:
            System.out.println(System.getProperty("user.dir")+"\n");
            String path="C:\\Users\\maiti\\OneDrive - UPV EHU\\Documentos\\EHES\\LAB\\heart-c.arff";

            //Artxiboa kargatu:
            DataSource source= new DataSource(path);
            Instances data=source.getDataSet();
            if (data.classIndex() == -1){
                data.setClassIndex(data.numAttributes() - 1);
            }

            //Fitxategiaren datuak lortu:
            fitxategiarenInformazioa(data, path);

            //Lehen 5 atributuen informazioa lortu:
            //lehen5AtributuenInfo(data);

        }catch (Exception e){System.out.println("Error: "+e.toString());}
    }
    public static void fitxategiarenInformazioa(Instances data, String path){
        //System.out.println("Klasearen indizea: "+data.classIndex());
        System.out.println("Fitxategiaren path-a: "+ path);
        System.out.println("Fitxategiak instantzia kopuru hau ditu: "+data.numInstances());
        System.out.println("Fitxategiak atributu kopuru hau ditu: "+data.numAttributes()+"\n");
        System.out.println(data.attribute(0).name()+", lehen atributuak, honako balio kopuru hau har ditzake (distinct): "
                + data.numDistinctValues(data.attribute(0))+"\n");

        //Klase atributuaren informazioa inprimatzeko:
        klaseAtributuarenInfo(data);

        System.out.println(data.attribute(data.classIndex()-1).name()+
                ", azken aurreko atributuak, honako missing balio kopuru hau ditu: "
                +data.attributeStats(data.classIndex()-1).missingCount);
        //1 PRAKTIKA GIDOIAREN 7. GALDERA (Lehen 5 atributuen informazioa atera)
        lehen5AtributuenInfo(data);

    }

    public static void klaseAtributuarenInfo(Instances data){
        System.out.println(data.attribute(data.numAttributes()-1).name()+
                ", azken atributuak, honako balio hauek har ditzake: ");
        //Hartu ditzaeen atributu balioak enumeratu
        Enumeration<Object> balioak=data.classAttribute().enumerateValues();
        int kont=0;
        int gutxien=0;
        //Banan banan errekorritu elementuak "gutxien" atributuan maiztasun txikiena gordez
        while(balioak.hasMoreElements()){
            Object n=balioak.nextElement();
            int maiztasuna= data.attributeStats(data.classIndex()).nominalCounts[kont++];

            if(kont==1){gutxien=maiztasuna;}
            else if(maiztasuna<gutxien){gutxien=maiztasuna;}

            System.out.println("     "+n+" balioa, maiztasun honekin atera da: "+maiztasuna);
        }

        //Maiztasun txikiena duten elementuak inprimatu:
        System.out.println("Gutxien atera diren atributuak ("+gutxien+" aldiz), hau da, klase minoritarioak: ");
        int k=0;
        balioak=data.classAttribute().enumerateValues();
        while (balioak.hasMoreElements()){
            Object n=balioak.nextElement();
            if(data.attributeStats(data.classIndex()).nominalCounts[k++]==gutxien){System.out.println("     "+n);}
        }
    }

    //Lehen 5 atributuen informazioa eskuratzeko (1 PRAKTIKA GIDOIAREN 7. GALDERA)
    public static void lehen5AtributuenInfo(Instances data){
        System.out.println("\nLehen 5 atributuen informazioa: ");
        for(int i=0; i<5; i++){
            System.out.println("Atributua: "+ data.attribute(i).name());

            //Zein motakoa den ateratzeko:
            boolean numerikoa=false;
            if(data.attribute(i).isNominal()){System.out.println("     Nominala da.");}
            else if(data.attribute(i).isString()){System.out.println("     String da.");}
            else if(data.attribute(i).isDate()){System.out.println("     Data da.");}
            else{System.out.println("     Numerikoa da.");numerikoa=true;}

            //Mota kontuan izan barik missing, distinct eta unique balioak dauden jakiteko:
            System.out.println("     Missing balio kopurua: " +data.attributeStats(i).missingCount);
            System.out.println("     Distinct balio kopurua: " +data.attributeStats(i).distinctCount);
            System.out.println("     Unique balio kopurua: " +data.attributeStats(i).uniqueCount);

            //Numerikoa bada atera beharreko datuak:
            if(numerikoa){
                System.out.println("     Min: "+data.attributeStats(i).numericStats.min);
                System.out.println("     Max: "+data.attributeStats(i).numericStats.max);
                System.out.println("     Batazbeste: "+data.attributeStats(i).numericStats.mean);
                System.out.println("     Desbiderapen: "+data.attributeStats(i).numericStats.stdDev);
            }
            System.out.println("\n");
        }
    }

    //BESTELAKOAK
    //Atributu guztiak inprimatzeko, haien estatistikekin:
    public static void atributuakInprimatu(Instances data){
        System.out.println("\nAtributuak honako hauek dira: \n");
        for(int i=0; i<data.numAttributes(); i++){
            System.out.println(data.attribute(i));
            System.out.println(data.attributeStats(i));
        }
    }
}