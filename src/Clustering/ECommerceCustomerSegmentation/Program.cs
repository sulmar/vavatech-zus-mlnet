// See https://aka.ms/new-console-template for more information

using System.Xml;
using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, Customer Segmentation!");

// Przykładowe dane
var customers = new List<CustomerData>()
{
    new CustomerData() { AveragePurchaseValue = 50, NumberOfPurchases = 5 },
    new CustomerData() { AveragePurchaseValue = 20, NumberOfPurchases = 2 },
    new CustomerData() { AveragePurchaseValue = 500, NumberOfPurchases = 50 },
    new CustomerData() { AveragePurchaseValue = 300, NumberOfPurchases = 30 },
    new CustomerData() { AveragePurchaseValue = 10, NumberOfPurchases = 1 },
    new CustomerData() { AveragePurchaseValue = 1000, NumberOfPurchases = 80 },
};

// 1. Utworzenie kontekstu
var context = new MLContext(seed: 1);    // dzięki Seed nasz model staje się deterministyczny

// 2. Wczytujemy dane
IDataView dataView = context.Data.LoadFromEnumerable(customers);

// 3. Przygotowanie trenera
var trainer = context.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3);

// 4. Tworzenie potoku przetwarzania
var pipeline = context.Transforms.Concatenate("Features", "AveragePurchaseValue", "NumberOfPurchases")
    .Append(trainer);

// 5. Trenowanie modelu
var model = pipeline.Fit(dataView);

// 6. Predykcja klastrow
var predictions = model.Transform(dataView);
var results = context.Data.CreateEnumerable<CustomerClusterPrediction>(predictions, reuseRowObject: false);

var clusterLabels = new Dictionary<uint, string>
{
    [1] = "Okazcjonalnie kupujący",
    [2] = "Bogaci klienci",
    [3] = "Lojalni klienci",
};

// 7. Wyświetlenie wynikw
int index = 0;
foreach (var (customer, prediction) in customers.Zip(results, (c, p) => (c, p)))
{
    Console.WriteLine($"Klient: {index}: Wartość zakupow: {customer.AveragePurchaseValue} Liczba transakcji: {customer.NumberOfPurchases} -> Klaster: {prediction.ClusterId} {clusterLabels[prediction.ClusterId]}");
    index++;
}

Console.WriteLine("Press any key to continue...");
Console.ReadKey();

public class CustomerData
{
    public float AveragePurchaseValue { get; set; } // średnia wartość zamowienia
    public float NumberOfPurchases { get; set; }    // Liczba transakcji
}

// Klasa definiuje wynik predykcji

public class CustomerClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint ClusterId { get; set; }
    
    [ColumnName("Score")]
    public float[] Distances { get; set; }
}
