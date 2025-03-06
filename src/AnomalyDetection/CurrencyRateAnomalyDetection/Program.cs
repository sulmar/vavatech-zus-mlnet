using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;

Console.WriteLine("Hello, Currency Rate Anomaly Detection!");

// dotnet add package Microsoft.ML

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();

// 2. Wczytujemy dane z pliku CSV
var rates = context.Data.LoadFromTextFile<CurrencyRateRawData>("archiwum_tab_a_2021.csv", hasHeader: true, separatorChar: ';');

var dataView = context.Data.SkipRows(rates, 1);

// Konwersja IDataView na List<T> 
List<CurrencyRateRawData> dataList = context.Data.CreateEnumerable<CurrencyRateRawData>(dataView, reuseRowObject: false).ToList();

// 3. Tworzenie pipeline'a z użyciem IidSpikeDetector

// dotnet add package Microsoft.ML.TimeSeries
var trainer = context.Transforms.DetectIidSpike(
    outputColumnName: "Prediction", 
    inputColumnName: "EUR", 
    confidence: 95d,  // Poziom ufności definiuje, jak "pewny" musi być model, aby uznać punkt za anomalię. Zakres 0..100 
    pvalueHistoryLength: dataList.Count / 2);


// Utworzenie mapowania daty
Action<CurrencyRateRawData, CurrencyRateData> mapper = (input, output) =>
{
    output.Date = DateTime.ParseExact(input.Date, "yyyyMMdd", CultureInfo.InvariantCulture);
    output.EUR = input.EUR;
};

// 4. Budowanie modelu
var pipeline = context.Transforms.CustomMapping(mapper, "DateMapper")
    .Append(trainer);

// 5. Trenowanie modelu
var model = pipeline.Fit(dataView);

// 6. Przekształcenie danych i predykcja
var predictions = model.Transform(dataView);


// 7. Pobranie wyników predykcji
var predictionResults = context.Data.CreateEnumerable<CurrencyRatePrediction>(predictions, reuseRowObject: false);


// 8. Wyświetlenie wyników
Console.WriteLine("Wyniki detekcji anomalii (IidSpikeDetector):");
int i = 0;
foreach (var (input, prediction) in dataList.Zip(predictionResults, (d, p) => (d, p)))
{
    // Wektor Prediction zawiera 3 wartości: [Alert, P-Value, Score]
    if (prediction.Alert)
        Console.BackgroundColor = ConsoleColor.Red;

    Console.WriteLine($"Przykład {i + 1}: Date = {input.Date:yyyy-MM-dd}, PValue = {prediction.PValue:F2}, Score = {prediction.Score:F2}, Alert = {prediction.Alert}");
    i++;

    Console.ResetColor();
}

Console.ReadLine();

public class CurrencyRateRawData
{
    [LoadColumn(0)]
    public string Date { get; set; }

    [LoadColumn(8)]
    public float EUR { get; set; }
}

public class CurrencyRateData
{
    public DateTime Date { get; set; }
    public float EUR { get; set; }
}


public class CurrencyRatePrediction
{
    // Wynik z IidSpikeDetector to wektor 3-wymiarowy: [Alert, P-Value, Score]
    [VectorType(3)]
    public double[] Prediction { get; set; }

    // Wygodne właściwości do odczytu wyników
    public double PValue => Prediction[1];
    public double Score => Prediction[2];
    public bool Alert => Prediction[0] == 1;
}