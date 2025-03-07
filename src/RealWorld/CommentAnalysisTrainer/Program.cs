// See https://aka.ms/new-console-template for more information

using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, Comments Analysis Trainer!");

// Tworzymy kontekst
var context = new MLContext();

// Ladowanie danych
var data = context.Data.LoadFromTextFile<CommentData>("training-data.csv", hasHeader: true, separatorChar: ',');

// Przygotowanie trenera
var trainer = context.BinaryClassification.Trainers.SdcaLogisticRegression();

// Tworzymy potok przetwarzania
var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(CommentData.CommentText))
    .Append(trainer);

// Trenowanie
var model = pipeline.Fit(data);

// Zapis modelu

var filepath = "sentiment-model.zip";

context.Model.Save(model, data.Schema, filepath);

Console.WriteLine($"Training complete and saved to disk {filepath}.");

public class CommentData
{
    [LoadColumn(0)]
    public string CommentText { get; set; }
    
    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment { get; set; }
}