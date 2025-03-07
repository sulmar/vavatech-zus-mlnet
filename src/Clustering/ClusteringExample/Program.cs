// See https://aka.ms/new-console-template for more information

using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, Clustering!");

// Przykładowe dane - dwa skupiska punktow
var data = new List<DataPoint>()
{
    new DataPoint() { Feature1 = 1.0f, Feature2 = 1.0f },
    new DataPoint() { Feature1 = 1.1f, Feature2 = 1.1f },
    new DataPoint() { Feature1 = 0.9f, Feature2 = 0.9f },
    
    new DataPoint() { Feature1 = 10.0f, Feature2 = 10.0f },
    new DataPoint() { Feature1 = 10.1f, Feature2 = 10.1f },
    new DataPoint() { Feature1 = 9.9f, Feature2 = 9.9f },
};

// 1. Tworzymy kontekst
var context = new MLContext(seed: 1);

// 2. Ladujemy dane
IDataView dataView = context.Data.LoadFromEnumerable(data); 

// 3. Przygotuwujemy trainera
var trainer = context.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 2);

// 4. Tworzymy potok 
var pipeline = context.Transforms
    .Concatenate("Features", "Feature1", "Feature2")
    .Append(trainer);

// 5. Trenujemy model
var model = pipeline.Fit(dataView);

// 6. Przetwarzamy dane przy użyciu wytrenowanego modelu
var predictions = model.Transform(dataView);
var results = context.Data.CreateEnumerable<ClusterPrediction>(predictions, reuseRowObject: false);

// 7. Wyświetlenie wynikow

int index = 0;
foreach (var prediction in results)
{
    Console.WriteLine($"Punkt danych {index}: Przydzielony klaster: {prediction.PredictedClusterId}");
    
    index++;
    
}

Console.ReadLine();

// Klasa reprezentująca pojedynczy punkt danych
public class DataPoint
{
    public float Feature1 { get; set; }
    public float Feature2 { get; set; }
}

// Klasa wynikowa predykcji klastrowania
public class ClusterPrediction
{
    // Przewidywany klaster (indeks klastrow zaczyna się od 1) 
    [ColumnName("PredictedLabel")]
    public uint PredictedClusterId { get; set; } 
    
    [ColumnName("Score")]
    public float[] Distances { get; set; } 
    
}