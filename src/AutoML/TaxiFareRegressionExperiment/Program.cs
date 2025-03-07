// See https://aka.ms/new-console-template for more information

using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, Taxi Fare Regression experiment!");

// dotnet add package Microsoft.ML
// dotnet add package Microsoft.ML.AutoML

// 1. Utworzymy kontekst
var context = new MLContext();

// 2. Wczytanie danych
string dataPath = "taxi-fare.csv";

IDataView dataView = context.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

// 3. Konfiguracja eksperymentu za pomocą AutoML
var experimentSettings = new RegressionExperimentSettings
{
    MaxExperimentTimeInSeconds = 60, // 60 sekund na eksperyment
    OptimizingMetric = RegressionMetric.RSquared // Optymalizacja R2
};

var pipeline = context.Auto().Regression(labelColumnName: "FareAmount");

// Tworzymy obsługę postępu
var progressHandler = new Progress<RunDetail<RegressionMetrics>>(progress =>
{
    if (progress.Exception != null)
        Console.WriteLine($"Błąd w modelu {progress.TrainerName} {progress.Exception.Message}");

    Console.WriteLine($"Model: {progress.TrainerName} Time: {progress.RuntimeInSeconds} seconds R2: {progress.ValidationMetrics?.RSquared:F4} RMSE: {progress.ValidationMetrics?.RootMeanSquaredError:F4} " );
});

// 4. Uruchomienie eksperymentu
Console.WriteLine("Rozpoczynam eksperyment AutoML...");
var experiment = context.Auto().CreateRegressionExperiment(experimentSettings);
var result = experiment.Execute(dataView, labelColumnName: "FareAmount", progressHandler: progressHandler);

// 5. Wyświetlenie najlepszych wynikow
Console.WriteLine($"Najlepszy model: {result.BestRun.TrainerName}");
Console.WriteLine($"R2: {result.BestRun.ValidationMetrics.RSquared}");

// 6. Predykcja na przykładowych danych
var sampleTrip = new TaxiTrip
{
    VendorId = "1",
    RateCode = 1,
    PassengerCount = 1,
    TripDistance = 3.5f
};

// Pobieramy najlepszy model
var model = result.BestRun.Model;

var predictionEngine = context.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

var prediction = predictionEngine.Predict(sampleTrip);

Console.WriteLine($"Przewidywana cena: {prediction.FareAmount:C2}");

Console.ReadKey();


    // vendor_id,rate_code,passenger_count,trip_distance,fare_amount
public class TaxiTrip
{
    [LoadColumn(0)]
    public string VendorId { get; set; }
    
    [LoadColumn(1)]
    public float RateCode { get; set; }

    [LoadColumn(2)]
    public float PassengerCount { get; set; }
    
    [LoadColumn(3)]
    public float TripDistance { get; set; }

    [LoadColumn(4)]
    public float FareAmount { get; set; }
    
}

// Klasa predykcji
public class TaxiTripFarePrediction
{
    [ColumnName("Score")]
    public float FareAmount { get; set; }
}
    
