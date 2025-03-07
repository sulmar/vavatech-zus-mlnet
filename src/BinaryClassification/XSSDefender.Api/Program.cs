using Microsoft.ML;
using XSSDefender.Api.Controllers;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();


// Rejestracja ML.NET w DI
builder.Services.AddSingleton<MLContext>(new MLContext());
builder.Services.AddSingleton<ITransformer>(serviceProvider =>
{
    var mlContext = serviceProvider.GetRequiredService<MLContext>();
    return mlContext.Model.Load("xss-model.zip", out var _);
});
builder.Services.AddSingleton<PredictionEngine<XssInput, XssPrediction>>(serviceProvider =>
{
    var mlContext = serviceProvider.GetRequiredService<MLContext>();
    var model = serviceProvider.GetRequiredService<ITransformer>();
    return mlContext.Model.CreatePredictionEngine<XssInput, XssPrediction>(model);
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
