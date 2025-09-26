# PowerShell version of run script
Write-Host "=== Compiling Multilayer Perceptron Implementation ===" -ForegroundColor Yellow
Write-Host "Compiling main.cpp with headers: Complex.h, Matrix.h, MLP.h" -ForegroundColor Cyan

# Compile the program
$compileResult = Start-Process -FilePath "g++" -ArgumentList @("-std=c++17", "-Wall", "-Wextra", "-O2", "-I.", "main.cpp", "-o", "mlp_train") -Wait -PassThru

# Check if compilation was successful
if ($compileResult.ExitCode -eq 0) {
    Write-Host "Compilation successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "=== Running MLP Training Program ===" -ForegroundColor Yellow
    Write-Host ""
    
    # Run the compiled program
    $runResult = Start-Process -FilePath ".\mlp_train.exe" -Wait -PassThru -NoNewWindow
    
    # Check if execution was successful
    if ($runResult.ExitCode -eq 0) {
        Write-Host ""
        Write-Host "=== Program completed successfully! ===" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "=== Program execution failed! ===" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Compilation failed!" -ForegroundColor Red
    exit 1
}
