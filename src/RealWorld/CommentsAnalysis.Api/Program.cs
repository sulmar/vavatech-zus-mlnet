using Microsoft.ML;
using Microsoft.ML.Data;

var builder = WebApplication.CreateBuilder(args);

// Rejestracja usług do obsługi ML.NET
builder.Services.AddSingleton<MLContext>();
builder.Services.AddSingleton<ITransformer>(serviceProvider =>
{
    var context = serviceProvider.GetRequiredService<MLContext>();

    return context.Model.Load("sentiment-model.zip", out var _);
});

builder.Services.AddSingleton<PredictionEngine<CommentData, SentimentPrediction>>(serviceProvider =>
{
    var context = serviceProvider.GetRequiredService<MLContext>();
    var model = serviceProvider.GetRequiredService<ITransformer>();
    return context.Model.CreatePredictionEngine<CommentData, SentimentPrediction>(model);
});

var app = builder.Build();

app.MapPost("/analyze-comment", (CommentData comment, PredictionEngine<CommentData, SentimentPrediction> predictionEngine) =>
{
    var prediction = predictionEngine.Predict(comment);
    
    return Results.Ok(new
    {
        Comment = comment.CommentText,
        IsPositive = prediction.Prediction,
    });
});

app.Run();

public class CommentData
{
    public string CommentText { get; set; }
}

public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
}