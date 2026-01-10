"""
Utility script to set up training data
Creates sample TypeScript files if data directory is empty
"""
import os
import argparse


def create_sample_data(data_dir: str, num_files: int = 50):
    """Create sample TypeScript files for training"""
    os.makedirs(data_dir, exist_ok=True)
    
    sample_code = [
        # Basic functions
        """function add(a: number, b: number): number {
    return a + b;
}""",
        """function multiply(x: number, y: number): number {
    return x * y;
}""",
        """function greet(name: string): string {
    return `Hello, ${name}!`;
}""",
        
        # Classes
        """class Calculator {
    private value: number = 0;
    
    add(n: number): void {
        this.value += n;
    }
    
    getValue(): number {
        return this.value;
    }
}""",
        
        # Interfaces
        """interface User {
    id: number;
    name: string;
    email: string;
}""",
        
        # Arrays and loops
        """function sumArray(arr: number[]): number {
    let sum = 0;
    for (const num of arr) {
        sum += num;
    }
    return sum;
}""",
        
        # Async/await
        """async function fetchData(url: string): Promise<any> {
    const response = await fetch(url);
    return await response.json();
}""",
        
        # Generics
        """function identity<T>(arg: T): T {
    return arg;
}""",
        
        # Type guards
        """function isString(value: unknown): value is string {
    return typeof value === 'string';
}""",
        
        # Error handling
        """function safeDivide(a: number, b: number): number | null {
    if (b === 0) {
        return null;
    }
    return a / b;
}"""
    ]
    
    # Create files
    for i in range(num_files):
        code = sample_code[i % len(sample_code)]
        # Add some variation
        code = code.replace('add', f'add{i % 3}')
        code = code.replace('Calculator', f'Calculator{i % 5}')
        
        file_path = os.path.join(data_dir, f'sample_{i:04d}.ts')
        with open(file_path, 'w') as f:
            f.write(code)
    
    print(f"Created {num_files} sample TypeScript files in {data_dir}")


def main():
    parser = argparse.ArgumentParser(description='Set up training data')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to create sample data')
    parser.add_argument('--num_files', type=int, default=50,
                        help='Number of sample files to create')
    
    args = parser.parse_args()
    
    create_sample_data(args.data_dir, args.num_files)


if __name__ == '__main__':
    main()
