// See https://aka.ms/new-console-template for more information

using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace XSSDetection
{
    public class XSSInput
    {
        [LoadColumn(0)] public int Id { get; set; }
        [LoadColumn(1)] public string Sentence { get; set; }
        [LoadColumn(2)] public bool Label { get; set; }
    }

    public class QueryPrediction
    {
        [ColumnName("PredictedLabel")] public bool IsSuspicious { get; set; }
        public float Score { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            MLContext conext = new MLContext(seed:1);

            // Wczytaj dane z pliku
            string dataPath = "XSS_dataset.csv";
            IDataView dataView = conext.Data.LoadFromTextFile(
                dataPath, hasHeader: true, separatorChar: ',', allowQuoting: true);

            var preview = dataView.Preview();

            // Tworzenie pipeline
            var dataProcessPipeline = conext.Transforms.Text.FeaturizeText(
                    outputColumnName: "Features", inputColumnName: nameof(XSSInput.Sentence))
                .Append(conext.Transforms.NormalizeMinMax("Features"))
                .Append(conext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label", featureColumnName: "Features"));

            // Trenowanie modelu
            Console.WriteLine("Rozpoczynam trenowanie modelu...");
            var model = dataProcessPipeline.Fit(dataView);

            // Ocena modelu
            Console.WriteLine("Ocena modelu...");
            var testData = conext.Data.LoadFromTextFile<XSSInput>(
                dataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(testData);

            var metrics = conext.BinaryClassification.Evaluate(
                predictions, labelColumnName: "Label");

            Console.WriteLine($"Dokładność: {metrics.Accuracy:P2}");
            Console.WriteLine($"Precyzja: {metrics.PositivePrecision:P2}");
            Console.WriteLine($"Czułość: {metrics.PositiveRecall:P2}");

            // Przykładowa predykcja
            Console.WriteLine("Przykładowa predykcja...");
            var predictionEngine = conext.Model.CreatePredictionEngine<XSSInput, QueryPrediction>(model);

            var example = new XSSInput { Sentence = "<script>alert('XSS')</script>" };
            var result = predictionEngine.Predict(example);

            Console.WriteLine($"Zapytanie: {example.Sentence}");
            Console.WriteLine($"Podejrzane: {(result.IsSuspicious ? "Tak" : "Nie")}, Wynik: {result.Score}");
        }
    }
}