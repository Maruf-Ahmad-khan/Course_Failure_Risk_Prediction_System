async function submitForm() {

    let payload = {
        Age: parseFloat(document.getElementById("Age").value),
        Gender: document.getElementById("Gender").value,
        City: document.getElementById("City").value,
        Highest_Qualification: document.getElementById("Highest_Qualification").value,
        Stream: document.getElementById("Stream").value,
        Year_Of_Completion: parseInt(document.getElementById("Year_Of_Completion").value),
        Are_you_currently_working: document.getElementById("Are_you_currently_working").value,
        Your_Designation: document.getElementById("Your_Designation").value,
        Employment_Type: document.getElementById("Employment_Type").value,
        First_Name: document.getElementById("First_Name").value || null,
        Last_Name: document.getElementById("Last_Name").value || null,
        Company_Name: document.getElementById("Company_Name").value || null
    };

    const res = await fetch("/predict", {
        method:"POST",
        headers:{ "Content-Type":"application/json"},
        body:JSON.stringify(payload)
    });

    let data = await res.json();

    document.getElementById("predictionValue").innerHTML = data.prediction;

    drawChart(data.prediction);
}


// ---------------- Forecast Line Graph ------------------

let chart;

function drawChart(predValue){

    const ctx = document.getElementById("forecastChart").getContext("2d");

    let fakeTrend = [
        predValue - 5,
        predValue - 2,
        predValue,
        predValue + 3,
        predValue + 6
    ];

    if(chart){
        chart.destroy();
    }

    chart = new Chart(ctx,{
        type:"line",
        data:{
            labels:["Previous","Old","Current","Future","Forecast"],
            datasets:[{
                label:"Risk Trend",
                data:fakeTrend,
                borderWidth:2,
            }]
        }
    });
}
