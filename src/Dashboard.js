import React from "react";
import { Line } from "react-chartjs-2";
import "chart.js/auto";

const data = {
  labels: ["1", "2", "3", "4", "5"],
  datasets: [
    {
      label: "Demo Chart",
      data: [10, 15, 7, 12, 8],
      borderColor: "#36a2eb",
      backgroundColor: "rgba(54, 162, 235, 0.2)",
    },
  ],
};

const Dashboard = () => {
  return (
    <div style={{ backgroundColor: "#121212", height: "100vh", padding: "20px", color: "white", textAlign: "center" }}>
      <h1>Modern React Dashboard</h1>
      <Line data={data} />
      <button style={{
        background: "linear-gradient(145deg, #1e1e1e, #2a2a2a)",
        border: "none", color: "white", padding: "10px 20px",
        borderRadius: "10px", boxShadow: "3px 3px 10px rgba(0,0,0,0.5)",
        marginTop: "20px"
      }}>
        Click Me
      </button>
    </div>
  );
};

export default Dashboard;
