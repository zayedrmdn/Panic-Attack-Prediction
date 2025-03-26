# Panic Attack Detector 🏥  

A simple web-based **Panic Attack Detector** that predicts panic disorder based on user input.  
This guide will walk you through setting up and running the application **without any coding knowledge**.  

---

## 🚀 Getting Started

Follow these steps to **download, set up, and run the app** easily.

### **1️⃣ Clone the Repository (Using GitHub Desktop)**
1. **Open [GitHub Desktop](https://desktop.github.com/)**  
2. Click **"File" > "Clone repository"**  
3. Select the **"GitHub.com"** tab  
4. Find this repository and click **"Clone"**  
5. Choose a folder to save the project and click **"Clone"**

Once finished, the project will be **downloaded to your computer**.

---

### **2️⃣ Set Up a Virtual Environment**
A virtual environment keeps the required dependencies separate from your system.

#### **🖥️ Windows**
1. Open **Command Prompt (`cmd`)** and navigate to the project folder:
   ```sh
   cd path\to\your\cloned\repo
   ```
2. Run this command to create a virtual environment:
   ```sh
   python -m venv venv
   ```
3. **Activate the virtual environment**:
   ```sh
   venv\Scripts\activate
   ```
   You should now see `(venv)` before your command line, meaning it's activated.

#### **🍏 macOS / Linux**
1. Open **Terminal** and navigate to the project folder:
   ```sh
   cd path/to/your/cloned/repo
   ```
2. Create the virtual environment:
   ```sh
   python3 -m venv venv
   ```
3. **Activate it**:
   ```sh
   source venv/bin/activate
   ```

---

### **3️⃣ Install Dependencies**
Now that the virtual environment is active, install all required packages:

```sh
pip install -r requirements.txt
```

This will **automatically install** everything needed to run the application.

---

### **4️⃣ Run the Web App**
After installation, start the application by running:

```sh
python app/main.py
```

You should see output like this:

```
 * Running on http://127.0.0.1:5000/
```

📌 **Now, open your web browser and go to** 👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)  

You can now **use the Panic Attack Detector!** 🎉  

---

### **5️⃣ Deactivating the Virtual Environment**
When you're done using the app, deactivate the virtual environment:  

#### **🖥️ Windows**
```sh
deactivate
```

#### **🍏 macOS / Linux**
```sh
deactivate
```

---

## 🛠️ **Troubleshooting**
### ❓ `pip` command not found?
Try using:
```sh
python -m pip install -r requirements.txt
```

### ❓ Can't run the app?
Ensure the virtual environment is **activated**, then run:
```sh
python app/main.py
```

---

## 📌 **Summary**
1. **Clone the repository using GitHub Desktop**  
2. **Set up a virtual environment** (`venv`)  
3. **Install dependencies** (`pip install -r requirements.txt`)  
4. **Run the app** (`python app/main.py`)  
5. **Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser**  

Now you're ready to use the **Panic Attack Detector!** 🚀😊

