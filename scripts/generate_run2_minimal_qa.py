#!/usr/bin/env python3
"""
Generate Phase 3a Run 2 Dataset: Minimal Q&A Obedience

Goals:
1. Short questions → 1 sentence answers
2. Length-bounded answers
3. "I don't know" for unknowable questions
4. Basic constraint following (same Q, different length targets)

All responses are 1-2 sentences max.
Target: 500+ episodes
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

SYSTEM_PROMPT = "You are MyPT, a helpful assistant."

def generate_qa_pairs() -> List[Tuple[str, str, str]]:
    """Generate Q&A pairs as (question, answer, category)."""
    pairs = []
    
    # === FACTUAL Q&A (1 sentence answers) - TECHNOLOGY ===
    tech_factual = [
        ("What is Python?", "Python is a high-level programming language known for its readability and versatility."),
        ("What is JavaScript?", "JavaScript is a programming language primarily used for web development."),
        ("What is an API?", "An API is a set of protocols that allows different software applications to communicate."),
        ("What is a database?", "A database is an organized collection of structured data stored electronically."),
        ("What is HTML?", "HTML is a markup language used to structure content on web pages."),
        ("What is CSS?", "CSS is a stylesheet language used to control the visual presentation of web pages."),
        ("What is a server?", "A server is a computer that provides services or resources to other computers over a network."),
        ("What is encryption?", "Encryption is the process of converting data into a coded format to prevent unauthorized access."),
        ("What is a firewall?", "A firewall is a security system that monitors and controls incoming and outgoing network traffic."),
        ("What is cloud computing?", "Cloud computing is the delivery of computing services over the internet."),
        ("What is machine learning?", "Machine learning is a type of AI that enables systems to learn from data without explicit programming."),
        ("What is a neural network?", "A neural network is a computing system inspired by biological neural networks in the brain."),
        ("What is Git?", "Git is a distributed version control system for tracking changes in source code."),
        ("What is Docker?", "Docker is a platform for developing, shipping, and running applications in containers."),
        ("What is Linux?", "Linux is an open-source operating system based on Unix."),
        ("What is a VPN?", "A VPN is a service that creates a secure, encrypted connection over the internet."),
        ("What is RAM?", "RAM is temporary memory that computers use to store data being actively used."),
        ("What is a CPU?", "A CPU is the primary processor that executes instructions in a computer."),
        ("What is an SSD?", "An SSD is a storage device that uses flash memory to store data persistently."),
        ("What is Kubernetes?", "Kubernetes is an open-source platform for automating deployment and management of containerized applications."),
        ("What is TypeScript?", "TypeScript is a typed superset of JavaScript that compiles to plain JavaScript."),
        ("What is React?", "React is a JavaScript library for building user interfaces."),
        ("What is Node.js?", "Node.js is a runtime environment that executes JavaScript outside a web browser."),
        ("What is SQL?", "SQL is a language for managing and querying relational databases."),
        ("What is NoSQL?", "NoSQL refers to non-relational databases designed for flexible data models."),
        ("What is MongoDB?", "MongoDB is a document-oriented NoSQL database."),
        ("What is PostgreSQL?", "PostgreSQL is an advanced open-source relational database."),
        ("What is Redis?", "Redis is an in-memory data structure store used as a database and cache."),
        ("What is GraphQL?", "GraphQL is a query language for APIs that allows clients to request specific data."),
        ("What is REST?", "REST is an architectural style for designing networked applications using HTTP."),
        ("What is WebSocket?", "WebSocket is a protocol for full-duplex communication over a single TCP connection."),
        ("What is OAuth?", "OAuth is an authorization protocol that allows third-party access without sharing passwords."),
        ("What is JWT?", "JWT is a compact token format for securely transmitting information between parties."),
        ("What is HTTPS?", "HTTPS is HTTP with encryption for secure communication over a network."),
        ("What is TLS?", "TLS is a cryptographic protocol for secure communication over networks."),
        ("What is SSH?", "SSH is a protocol for secure remote login and command execution."),
        ("What is DNS?", "DNS translates domain names into IP addresses."),
        ("What is TCP/IP?", "TCP/IP is the fundamental protocol suite for internet communication."),
        ("What is JSON?", "JSON is a lightweight data interchange format based on JavaScript syntax."),
        ("What is XML?", "XML is a markup language for encoding documents in a human and machine-readable format."),
        ("What is YAML?", "YAML is a human-readable data serialization format often used for configuration files."),
        ("What is Markdown?", "Markdown is a lightweight markup language for creating formatted text."),
        ("What is a microservice?", "A microservice is a small, independent service that performs a specific function."),
        ("What is serverless computing?", "Serverless computing lets you run code without managing servers."),
        ("What is CI/CD?", "CI/CD is a practice of automating code integration, testing, and deployment."),
        ("What is DevOps?", "DevOps is a set of practices combining software development and IT operations."),
        ("What is Agile?", "Agile is a project management methodology emphasizing iterative development and collaboration."),
        ("What is Scrum?", "Scrum is an Agile framework for managing and completing complex projects."),
        ("What is a sprint?", "A sprint is a fixed time period in Scrum during which specific work is completed."),
        ("What is refactoring?", "Refactoring is restructuring existing code without changing its external behavior."),
    ]
    for q, a in tech_factual:
        pairs.append((q, a, "factual_tech"))
    
    # === FACTUAL Q&A - SCIENCE ===
    science_factual = [
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight into chemical energy."),
        ("What is gravity?", "Gravity is a force that attracts objects with mass toward each other."),
        ("What is DNA?", "DNA is a molecule that carries genetic instructions for living organisms."),
        ("What is an atom?", "An atom is the smallest unit of matter that retains the properties of an element."),
        ("What is evolution?", "Evolution is the process by which species change over time through natural selection."),
        ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second in a vacuum."),
        ("What is a black hole?", "A black hole is a region of space where gravity is so strong that nothing can escape."),
        ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns."),
        ("What is a virus?", "A virus is a microscopic infectious agent that replicates inside living cells."),
        ("What is an ecosystem?", "An ecosystem is a community of living organisms interacting with their environment."),
        ("What is a neutron?", "A neutron is a subatomic particle with no electric charge found in atomic nuclei."),
        ("What is a proton?", "A proton is a positively charged subatomic particle found in atomic nuclei."),
        ("What is an electron?", "An electron is a negatively charged subatomic particle that orbits the nucleus."),
        ("What is a molecule?", "A molecule is a group of atoms bonded together."),
        ("What is a cell?", "A cell is the basic structural and functional unit of living organisms."),
        ("What is mitosis?", "Mitosis is the process of cell division that produces two identical daughter cells."),
        ("What is meiosis?", "Meiosis is cell division that produces four cells with half the original chromosome number."),
        ("What is a gene?", "A gene is a unit of heredity that codes for a specific trait."),
        ("What is a chromosome?", "A chromosome is a structure containing DNA and genetic information."),
        ("What is natural selection?", "Natural selection is the process where organisms better adapted to their environment survive and reproduce."),
        ("What is entropy?", "Entropy is a measure of disorder or randomness in a system."),
        ("What is kinetic energy?", "Kinetic energy is the energy an object possesses due to its motion."),
        ("What is potential energy?", "Potential energy is stored energy based on an object's position or state."),
        ("What is the periodic table?", "The periodic table is a chart organizing chemical elements by atomic number and properties."),
        ("What is an isotope?", "An isotope is a variant of an element with a different number of neutrons."),
        ("What is a catalyst?", "A catalyst is a substance that speeds up a chemical reaction without being consumed."),
        ("What is pH?", "pH is a scale measuring how acidic or basic a solution is."),
        ("What is a wavelength?", "A wavelength is the distance between successive peaks of a wave."),
        ("What is frequency?", "Frequency is the number of wave cycles that pass a point per unit time."),
        ("What is the electromagnetic spectrum?", "The electromagnetic spectrum is the range of all types of electromagnetic radiation."),
    ]
    for q, a in science_factual:
        pairs.append((q, a, "factual_science"))
    
    # === FACTUAL Q&A - GENERAL KNOWLEDGE ===
    general_factual = [
        ("What is democracy?", "Democracy is a system of government where citizens exercise power through voting."),
        ("What is economics?", "Economics is the study of how societies allocate scarce resources."),
        ("What is philosophy?", "Philosophy is the study of fundamental questions about existence, knowledge, and ethics."),
        ("What is psychology?", "Psychology is the scientific study of mind and behavior."),
        ("What is sociology?", "Sociology is the study of human society and social relationships."),
        ("What is a contract?", "A contract is a legally binding agreement between two or more parties."),
        ("What is inflation?", "Inflation is the rate at which prices for goods and services increase over time."),
        ("What is a patent?", "A patent is a legal right that grants exclusive ownership of an invention."),
        ("What is copyright?", "Copyright is legal protection for original creative works."),
        ("What is a trademark?", "A trademark is a symbol or name legally registered to represent a company or product."),
        ("What is capitalism?", "Capitalism is an economic system based on private ownership and free markets."),
        ("What is socialism?", "Socialism is an economic system where the means of production are collectively owned."),
        ("What is a recession?", "A recession is a significant decline in economic activity lasting months or longer."),
        ("What is GDP?", "GDP is the total value of goods and services produced in a country."),
        ("What is supply and demand?", "Supply and demand is an economic model explaining how prices are determined in markets."),
        ("What is a monopoly?", "A monopoly is a market structure where a single seller dominates."),
        ("What is bankruptcy?", "Bankruptcy is a legal process for individuals or businesses unable to pay debts."),
        ("What is liability?", "Liability is legal responsibility for one's actions or debts."),
        ("What is equity?", "Equity is ownership interest in an asset after subtracting liabilities."),
        ("What is due diligence?", "Due diligence is the investigation of a business before a transaction."),
        ("What is an audit?", "An audit is an official examination of accounts or processes."),
        ("What is compliance?", "Compliance is adherence to laws, regulations, and standards."),
        ("What is governance?", "Governance is the system of rules and practices for directing an organization."),
        ("What is ethics?", "Ethics is the branch of philosophy dealing with moral principles."),
        ("What is a stakeholder?", "A stakeholder is anyone with an interest in an organization's success."),
        ("What is a hypothesis?", "A hypothesis is a proposed explanation that can be tested."),
        ("What is empirical evidence?", "Empirical evidence is information obtained through observation or experimentation."),
        ("What is a variable?", "A variable is a factor that can change in an experiment or equation."),
        ("What is correlation?", "Correlation is a statistical relationship between two variables."),
        ("What is causation?", "Causation means one event directly causes another to happen."),
    ]
    for q, a in general_factual:
        pairs.append((q, a, "factual_general"))
    
    # === WITH EXPLICIT LENGTH CONSTRAINTS ===
    constrained = [
        # "In one sentence"
        ("What is Python? Answer in one sentence.", "Python is a versatile, high-level programming language."),
        ("What is a database? One sentence only.", "A database is an organized collection of data."),
        ("Explain encryption in one sentence.", "Encryption converts data into unreadable code for security."),
        ("What is an API? Keep it to one sentence.", "An API allows different software systems to communicate."),
        ("Define machine learning in one sentence.", "Machine learning enables computers to learn patterns from data."),
        ("What is Docker? One sentence.", "Docker packages applications into isolated containers."),
        ("Explain Git in one sentence.", "Git tracks changes in source code for version control."),
        ("What is a firewall? One sentence answer.", "A firewall monitors and filters network traffic."),
        ("Define cloud computing in one sentence.", "Cloud computing delivers computing resources over the internet."),
        ("What is a neural network? One sentence.", "A neural network is a computing system that learns patterns from data."),
        ("Explain REST in one sentence.", "REST is an architectural style for building web APIs."),
        ("What is SQL? One sentence only.", "SQL is a language for managing relational databases."),
        ("Define DevOps in one sentence.", "DevOps combines development and operations for faster delivery."),
        ("What is Agile? One sentence.", "Agile is an iterative approach to project management."),
        ("Explain OAuth in one sentence.", "OAuth is a protocol for delegated authorization."),
        
        # "Briefly"
        ("What is Git? Briefly.", "Git is a version control system for tracking code changes."),
        ("Explain cloud computing briefly.", "Cloud computing delivers computing services over the internet."),
        ("What is a firewall? Brief answer.", "A firewall monitors and controls network traffic for security."),
        ("Describe Docker briefly.", "Docker packages applications into portable containers."),
        ("What is Linux? Keep it brief.", "Linux is a free, open-source operating system."),
        ("Explain machine learning briefly.", "Machine learning lets systems learn from data automatically."),
        ("What is an API? Briefly.", "An API enables software applications to communicate."),
        ("Describe encryption briefly.", "Encryption converts data into secure, unreadable code."),
        ("What is a database? Brief answer.", "A database stores and organizes structured data."),
        ("Explain Kubernetes briefly.", "Kubernetes automates deployment of containerized applications."),
        ("What is React? Briefly.", "React is a JavaScript library for building user interfaces."),
        ("Describe TCP/IP briefly.", "TCP/IP is the core protocol suite for internet communication."),
        ("What is JSON? Brief answer.", "JSON is a lightweight data interchange format."),
        ("Explain microservices briefly.", "Microservices are small, independent services that work together."),
        ("What is CI/CD? Briefly.", "CI/CD automates code integration, testing, and deployment."),
        
        # "In X words"
        ("What is Python? Answer in 10 words or less.", "Python is a popular, easy-to-read programming language."),
        ("Define a server in under 15 words.", "A server is a computer that provides services to other computers."),
        ("Explain RAM in 10 words.", "RAM is temporary memory for active computer processes."),
        ("What is HTML in 10 words or less?", "HTML structures content displayed on web pages."),
        ("Define encryption in under 10 words.", "Encryption converts data into secure, unreadable code."),
        ("What is Git in 10 words?", "Git tracks code changes for version control."),
        ("Explain DNS in under 10 words.", "DNS translates domain names to IP addresses."),
        ("Define API in 10 words or less.", "An API lets software applications communicate."),
        ("What is a firewall in 10 words?", "A firewall monitors and controls network traffic."),
        ("Explain SQL in under 10 words.", "SQL queries and manages relational databases."),
        ("Define Docker in 10 words.", "Docker packages applications into portable containers."),
        ("What is CSS in 10 words?", "CSS controls the visual styling of web pages."),
        ("Explain OAuth in 10 words or less.", "OAuth allows secure third-party access without passwords."),
        ("Define REST in under 10 words.", "REST is an architecture for web APIs."),
        ("What is JSON in 10 words?", "JSON is a text format for data exchange."),
        
        # "Short answer"
        ("What is JavaScript? Short answer.", "JavaScript is a programming language for web development."),
        ("What is CSS? Short answer please.", "CSS styles the visual appearance of web pages."),
        ("What is a VPN? Give a short answer.", "A VPN creates secure, encrypted internet connections."),
        ("Explain Kubernetes shortly.", "Kubernetes automates container deployment and management."),
        ("What is a CPU? Short answer.", "A CPU executes instructions in a computer."),
        ("What is MongoDB? Short answer.", "MongoDB is a document-oriented NoSQL database."),
        ("Explain GraphQL shortly.", "GraphQL is a query language for APIs."),
        ("What is TypeScript? Short answer.", "TypeScript adds static typing to JavaScript."),
        ("What is Node.js? Short answer please.", "Node.js runs JavaScript outside browsers."),
        ("Explain WebSocket shortly.", "WebSocket enables real-time two-way communication."),
        ("What is Redis? Short answer.", "Redis is an in-memory data store."),
        ("What is PostgreSQL? Short answer.", "PostgreSQL is an advanced relational database."),
        ("Explain SSH shortly.", "SSH provides secure remote access to servers."),
        ("What is YAML? Short answer.", "YAML is a human-readable configuration format."),
        ("What is serverless? Short answer.", "Serverless runs code without managing servers."),
    ]
    for q, a in constrained:
        pairs.append((q, a, "constrained"))
    
    # === SAME QUESTION, DIFFERENT CONSTRAINTS ===
    variations = [
        # Python variations
        ("What is Python?", "Python is a high-level, interpreted programming language known for its clean syntax and versatility."),
        ("What is Python? One sentence.", "Python is a versatile programming language."),
        ("What is Python? 5 words.", "Popular, readable programming language."),
        
        # API variations
        ("What is an API?", "An API is a set of rules and protocols that allows different software applications to communicate with each other."),
        ("What is an API? Briefly.", "An API enables software systems to communicate."),
        ("What is an API? Maximum 8 words.", "Rules for software communication."),
        
        # Machine learning variations
        ("What is machine learning?", "Machine learning is a branch of artificial intelligence where systems learn patterns from data to make predictions."),
        ("What is machine learning? One sentence.", "Machine learning lets computers learn from data."),
        ("What is machine learning? 10 words max.", "AI that learns patterns from data automatically."),
        
        # Database variations
        ("What is a database?", "A database is an organized collection of structured data, typically stored and accessed electronically from a computer system."),
        ("What is a database? Short answer.", "A database stores organized data electronically."),
        ("What is a database? 6 words.", "Organized electronic data storage system."),
        
        # Encryption variations
        ("Explain encryption.", "Encryption is the process of converting readable data into an unreadable format using algorithms, ensuring only authorized parties can access the original information."),
        ("Explain encryption briefly.", "Encryption converts data into secure code."),
        ("Explain encryption in 5 words.", "Secures data through encoding."),
    ]
    for q, a in variations:
        pairs.append((q, a, "variation"))
    
    # === "I DON'T KNOW" RESPONSES ===
    unknowable = [
        # Personal/private information
        ("What is my password?", "I don't know your password and cannot access personal information."),
        ("What am I thinking right now?", "I cannot read minds or know your thoughts."),
        ("What will happen to me tomorrow?", "I cannot predict the future or know what will happen to you."),
        ("What is my name?", "I don't have access to your personal information unless you share it."),
        ("Where do I live?", "I don't know where you live."),
        ("What is my email address?", "I don't have access to your email address."),
        ("What did I eat yesterday?", "I don't know what you ate yesterday."),
        ("Who is my best friend?", "I don't have information about your personal relationships."),
        ("What is my phone number?", "I don't have access to your phone number."),
        ("What is my age?", "I don't know your age unless you tell me."),
        ("Where did I go to school?", "I don't have information about your education history."),
        ("What is my job?", "I don't know your occupation unless you share it."),
        ("What did I say earlier?", "I only have context from our current conversation."),
        ("What files are on my computer?", "I cannot access your local files or computer."),
        ("What is in my bank account?", "I cannot access your financial information."),
        
        # Unknowable facts
        ("What will the stock market do next week?", "I cannot predict future stock market movements."),
        ("Will it rain in Paris on March 15, 2030?", "I cannot predict specific future weather events."),
        ("Who will win the next World Cup?", "I cannot predict future sports outcomes."),
        ("What is the meaning of life?", "There is no universally agreed-upon answer to the meaning of life."),
        ("Is there a god?", "I cannot provide definitive answers about religious or metaphysical questions."),
        ("What happens after death?", "This is a philosophical question without a scientifically verifiable answer."),
        ("What will be invented in 2050?", "I cannot predict future inventions."),
        ("Will humans colonize Mars?", "I cannot predict future events with certainty."),
        ("Who will be president in 2040?", "I cannot predict future elections."),
        ("What lottery numbers will win?", "Lottery numbers are random and unpredictable."),
        ("Will AI take over the world?", "I cannot predict the future of AI development."),
        ("What will I dream about tonight?", "Dreams are unpredictable and I cannot foresee them."),
        ("Will my business succeed?", "I cannot predict business outcomes."),
        ("What is the best decision for me?", "I don't have enough context about your specific situation to advise."),
        ("Should I trust this person?", "I cannot make personal trust judgments for you."),
        
        # Outside knowledge cutoff
        ("What happened in the news yesterday?", "I don't have access to real-time news or current events."),
        ("What is the current price of Bitcoin?", "I don't have access to real-time market data."),
        ("What time is it right now?", "I don't have access to current time information."),
        ("What is the weather like today?", "I don't have access to current weather data."),
        ("What is the latest iPhone model?", "My knowledge has a cutoff date and may not include the latest products."),
        ("Who won the game last night?", "I don't have access to live sports results."),
        ("What is trending on social media?", "I cannot access real-time social media data."),
        ("Is this website currently down?", "I cannot check live website status."),
        ("What is the exchange rate today?", "I don't have access to live exchange rates."),
        ("How many COVID cases are there today?", "I don't have access to real-time health statistics."),
        
        # Nonsense/unanswerable
        ("What color is the number seven?", "Numbers don't have inherent colors; this question has no factual answer."),
        ("How much does a thought weigh?", "Thoughts are not physical objects and don't have weight."),
        ("What is north of north?", "North is a direction, not a location, so nothing is north of it."),
        ("What does silence sound like?", "Silence is the absence of sound, so it doesn't have a sound."),
        ("How long is a piece of string?", "This depends on the specific string in question."),
        ("What is the taste of blue?", "Colors don't have inherent tastes."),
        ("Can you draw a round square?", "A round square is a logical contradiction and cannot exist."),
        ("What is the sound of one hand clapping?", "This is a philosophical koan without a literal answer."),
        ("Where does the light go when you turn it off?", "Light doesn't go anywhere; photons are absorbed."),
        ("Can you create a rock so heavy you can't lift it?", "This is a paradox without a meaningful answer."),
    ]
    for q, a in unknowable:
        pairs.append((q, a, "unknowable"))
    
    # === SIMPLE INSTRUCTIONS ===
    instructions = [
        ("Tell me about Python.", "Python is a high-level programming language created by Guido van Rossum, widely used for web development, data science, and automation."),
        ("Explain what a firewall does.", "A firewall monitors network traffic and blocks unauthorized access based on security rules."),
        ("Describe machine learning.", "Machine learning is a subset of AI that enables systems to learn and improve from experience without explicit programming."),
        ("Give me information about Docker.", "Docker is a platform that uses containerization to package applications with their dependencies for consistent deployment."),
        ("Tell me about encryption.", "Encryption transforms readable data into an encoded format that can only be decoded with the correct key."),
        ("Explain APIs.", "APIs are interfaces that allow different software systems to communicate and share data with each other."),
        ("Describe what Git does.", "Git tracks changes in source code, enabling multiple developers to collaborate on projects efficiently."),
        ("Tell me about databases.", "Databases are organized systems for storing, managing, and retrieving structured data."),
        ("Explain cloud computing.", "Cloud computing provides on-demand access to computing resources like servers and storage over the internet."),
        ("Describe neural networks.", "Neural networks are computing systems modeled after the brain, used in machine learning to recognize patterns."),
        ("Tell me about Kubernetes.", "Kubernetes is an orchestration platform that automates deployment, scaling, and management of containerized applications."),
        ("Explain microservices.", "Microservices is an architecture where applications are built as small, independent services that communicate via APIs."),
        ("Describe version control.", "Version control tracks changes to files over time, allowing collaboration and the ability to revert to previous versions."),
        ("Tell me about REST APIs.", "REST APIs use HTTP methods to perform operations on resources identified by URLs."),
        ("Explain authentication.", "Authentication verifies the identity of users, typically through passwords, tokens, or biometrics."),
        ("Describe containerization.", "Containerization packages applications with their dependencies into isolated, portable units."),
        ("Tell me about load balancing.", "Load balancing distributes network traffic across multiple servers to ensure reliability and performance."),
        ("Explain caching.", "Caching stores frequently accessed data in fast storage to reduce latency and server load."),
        ("Describe CI/CD pipelines.", "CI/CD pipelines automate the process of testing, building, and deploying code changes."),
        ("Tell me about web servers.", "Web servers process HTTP requests and serve web pages or API responses to clients."),
        ("Explain DNS.", "DNS translates human-readable domain names into IP addresses that computers use to connect."),
        ("Describe SQL injection.", "SQL injection is an attack that inserts malicious SQL code through user input to manipulate databases."),
        ("Tell me about two-factor authentication.", "Two-factor authentication requires two different verification methods to confirm identity."),
        ("Explain CORS.", "CORS is a security feature that controls how web pages can request resources from other domains."),
        ("Describe the OSI model.", "The OSI model is a conceptual framework with seven layers that describes how network communication works."),
        ("Tell me about TCP handshake.", "The TCP handshake is a three-step process to establish a reliable connection between client and server."),
        ("Explain SSL certificates.", "SSL certificates encrypt connections and verify website identity to enable secure HTTPS communication."),
        ("Describe webhooks.", "Webhooks are automated messages sent from one system to another when a specific event occurs."),
        ("Tell me about OAuth.", "OAuth is an authorization protocol that allows third-party applications to access user data without exposing passwords."),
        ("Explain database indexing.", "Database indexing creates data structures that speed up data retrieval at the cost of additional storage."),
    ]
    for q, a in instructions:
        pairs.append((q, a, "instruction"))
    
    # === YES/NO WITH BRIEF EXPLANATION ===
    yes_no_explain = [
        ("Is Python free to use?", "Yes, Python is free and open-source software."),
        ("Is JavaScript the same as Java?", "No, JavaScript and Java are completely different programming languages."),
        ("Can a firewall prevent all cyber attacks?", "No, firewalls are one layer of security but cannot prevent all types of attacks."),
        ("Is machine learning the same as AI?", "No, machine learning is a subset of artificial intelligence."),
        ("Does encryption guarantee security?", "No, encryption improves security but no system is completely unbreakable."),
        ("Is Linux an operating system?", "Yes, Linux is a free, open-source operating system."),
        ("Can Git track changes in any file type?", "Yes, Git can track changes in any type of file."),
        ("Is HTML a programming language?", "No, HTML is a markup language, not a programming language."),
        ("Do databases require a server?", "Not always; some databases like SQLite are serverless."),
        ("Is cloud computing always cheaper?", "No, cloud computing costs depend on usage and can sometimes exceed on-premises costs."),
        ("Is CSS a programming language?", "No, CSS is a stylesheet language for styling."),
        ("Can Docker run Windows containers on Linux?", "No, Docker containers must match the host OS kernel."),
        ("Is TypeScript compiled?", "Yes, TypeScript compiles to JavaScript."),
        ("Can you run Python without installing it?", "Yes, some online interpreters allow running Python in a browser."),
        ("Is REST a protocol?", "No, REST is an architectural style, not a protocol."),
        ("Does MongoDB use SQL?", "No, MongoDB uses its own query language, not SQL."),
        ("Is JSON binary?", "No, JSON is a text-based format."),
        ("Can Kubernetes run without Docker?", "Yes, Kubernetes supports multiple container runtimes."),
        ("Is SSH secure?", "Yes, SSH uses encryption for secure communication."),
        ("Does HTTPS require a certificate?", "Yes, HTTPS requires an SSL/TLS certificate."),
        ("Is GraphQL a database?", "No, GraphQL is a query language for APIs."),
        ("Can a VPN hide your IP address?", "Yes, a VPN masks your real IP address."),
        ("Is 0 a positive number?", "No, zero is neither positive nor negative."),
        ("Is YAML valid JSON?", "No, YAML and JSON have different syntaxes but similar purposes."),
        ("Can RAM store data permanently?", "No, RAM loses data when power is off."),
        ("Is an SSD faster than an HDD?", "Yes, SSDs are significantly faster than traditional HDDs."),
        ("Does TCP guarantee delivery?", "Yes, TCP ensures reliable, ordered delivery."),
        ("Is UDP reliable?", "No, UDP does not guarantee delivery or order."),
        ("Can JavaScript run on servers?", "Yes, Node.js allows JavaScript to run server-side."),
        ("Is Git the same as GitHub?", "No, Git is version control software, GitHub is a hosting platform."),
    ]
    for q, a in yes_no_explain:
        pairs.append((q, a, "yes_no"))
    
    # === COMPARISONS (brief) ===
    comparisons = [
        ("What is the difference between Python and Java?", "Python emphasizes readability and simplicity, while Java focuses on portability and strong typing."),
        ("How does SQL differ from NoSQL?", "SQL databases use structured tables with fixed schemas, while NoSQL databases are more flexible and schema-less."),
        ("What is the difference between HTTP and HTTPS?", "HTTPS is HTTP with encryption, making data transfer secure."),
        ("Compare RAM and storage.", "RAM is fast temporary memory for active processes, while storage is slower but permanent."),
        ("What is the difference between a compiler and interpreter?", "A compiler translates all code before execution, while an interpreter translates line by line during execution."),
        ("How do containers differ from virtual machines?", "Containers share the host OS kernel and are lighter, while VMs include a full OS and are more isolated."),
        ("Compare public and private clouds.", "Public clouds are shared infrastructure, while private clouds are dedicated to a single organization."),
        ("What is the difference between TCP and UDP?", "TCP ensures reliable, ordered delivery, while UDP is faster but doesn't guarantee delivery."),
        ("How does symmetric differ from asymmetric encryption?", "Symmetric uses one key for both encryption and decryption, asymmetric uses a key pair."),
        ("Compare Git and SVN.", "Git is distributed with local repositories, while SVN is centralized with a single server."),
        ("What is the difference between REST and GraphQL?", "REST uses fixed endpoints, while GraphQL lets clients request specific data."),
        ("Compare MongoDB and PostgreSQL.", "MongoDB stores documents flexibly, while PostgreSQL is a structured relational database."),
        ("How does React differ from Angular?", "React is a library focused on UI, while Angular is a full framework with more built-in features."),
        ("What is the difference between frontend and backend?", "Frontend handles user interface, while backend handles data processing and storage."),
        ("Compare CPU and GPU.", "CPUs handle general tasks sequentially, while GPUs process many parallel tasks simultaneously."),
        ("How does authentication differ from authorization?", "Authentication verifies identity, while authorization determines what actions are permitted."),
        ("What is the difference between cookies and sessions?", "Cookies are stored client-side, while sessions are stored server-side."),
        ("Compare IPv4 and IPv6.", "IPv4 uses 32-bit addresses, while IPv6 uses 128-bit addresses for more available addresses."),
        ("What is the difference between stack and queue?", "A stack is LIFO (last in, first out), while a queue is FIFO (first in, first out)."),
        ("Compare arrays and linked lists.", "Arrays have fixed size and fast access, while linked lists have dynamic size but slower access."),
        ("What is the difference between GET and POST?", "GET requests data and is visible in URL, while POST sends data in the request body."),
        ("Compare monolithic and microservices architecture.", "Monolithic is one unified codebase, while microservices are independent, smaller services."),
        ("How does JavaScript differ from TypeScript?", "TypeScript adds static typing and compiles to JavaScript."),
        ("What is the difference between class and object?", "A class is a blueprint, while an object is an instance of that class."),
        ("Compare synchronous and asynchronous operations.", "Synchronous blocks until complete, while asynchronous allows other operations during execution."),
        ("What is the difference between unit and integration testing?", "Unit tests check individual components, while integration tests check how components work together."),
        ("Compare stateless and stateful applications.", "Stateless apps don't retain data between requests, while stateful apps maintain session data."),
        ("How does SSD differ from HDD?", "SSDs use flash memory and are faster, while HDDs use spinning disks and are cheaper."),
        ("What is the difference between hot and cold storage?", "Hot storage is for frequently accessed data, cold storage is for archival data."),
        ("Compare eager and lazy loading.", "Eager loading fetches all data upfront, lazy loading fetches only when needed."),
    ]
    for q, a in comparisons:
        pairs.append((q, a, "comparison"))
    
    # === HOW-TO (very brief) ===
    howto = [
        ("How do I create a variable in Python?", "Use the syntax: variable_name = value"),
        ("How do I print to console in JavaScript?", "Use console.log('your message')"),
        ("How do I comment code in Python?", "Use the # symbol at the start of the line."),
        ("How do I start a Git repository?", "Run 'git init' in your project folder."),
        ("How do I install a Python package?", "Use 'pip install package_name' in your terminal."),
        ("How do I create a Docker container?", "Use 'docker run image_name' to create and start a container."),
        ("How do I check my Git status?", "Run 'git status' to see current changes."),
        ("How do I connect to a database in Python?", "Use a library like sqlite3 or SQLAlchemy with a connection string."),
        ("How do I create a function in JavaScript?", "Use 'function name() { }' or arrow syntax '() => { }'."),
        ("How do I exit Vim?", "Press Escape, then type :q! to quit without saving or :wq to save and quit."),
        ("How do I commit in Git?", "Run 'git commit -m \"your message\"'."),
        ("How do I push to Git remote?", "Run 'git push origin branch_name'."),
        ("How do I pull from Git?", "Run 'git pull origin branch_name'."),
        ("How do I create a branch in Git?", "Run 'git checkout -b branch_name'."),
        ("How do I merge branches in Git?", "Checkout the target branch, then run 'git merge source_branch'."),
        ("How do I undo the last Git commit?", "Run 'git reset --soft HEAD~1' to undo while keeping changes."),
        ("How do I list Docker containers?", "Run 'docker ps' to see running containers, 'docker ps -a' for all."),
        ("How do I stop a Docker container?", "Run 'docker stop container_id'."),
        ("How do I remove a Docker image?", "Run 'docker rmi image_name'."),
        ("How do I create a virtual environment in Python?", "Run 'python -m venv env_name'."),
        ("How do I activate a Python virtual environment?", "On Windows: env\\Scripts\\activate, on Unix: source env/bin/activate"),
        ("How do I read a file in Python?", "Use 'with open(\"filename\", \"r\") as f: content = f.read()'"),
        ("How do I write to a file in Python?", "Use 'with open(\"filename\", \"w\") as f: f.write(content)'"),
        ("How do I create a list in Python?", "Use square brackets: my_list = [1, 2, 3]"),
        ("How do I create a dictionary in Python?", "Use curly braces: my_dict = {\"key\": \"value\"}"),
        ("How do I loop through a list in Python?", "Use 'for item in my_list:' followed by your code."),
        ("How do I check if a key exists in a dict?", "Use 'if key in my_dict:' or my_dict.get(key)."),
        ("How do I import a module in Python?", "Use 'import module_name' or 'from module import function'."),
        ("How do I handle exceptions in Python?", "Use 'try: ... except ExceptionType: ...'"),
        ("How do I define a class in Python?", "Use 'class ClassName:' followed by methods."),
    ]
    for q, a in howto:
        pairs.append((q, a, "howto"))
    
    # === DEFINITION REQUESTS ===
    definitions = [
        ("Define recursion.", "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems."),
        ("Define latency.", "Latency is the delay between a request and its response."),
        ("Define bandwidth.", "Bandwidth is the maximum rate of data transfer across a network."),
        ("Define scalability.", "Scalability is a system's ability to handle increased load by adding resources."),
        ("Define authentication.", "Authentication is the process of verifying a user's identity."),
        ("Define authorization.", "Authorization determines what actions an authenticated user can perform."),
        ("Define idempotency.", "Idempotency means an operation produces the same result regardless of how many times it's executed."),
        ("Define polymorphism.", "Polymorphism allows objects of different types to be treated through a common interface."),
        ("Define abstraction.", "Abstraction hides complex implementation details behind a simpler interface."),
        ("Define encapsulation.", "Encapsulation bundles data and methods together while restricting direct access to internal state."),
        ("Define inheritance.", "Inheritance allows a class to derive properties and methods from a parent class."),
        ("Define composition.", "Composition is a design principle where objects contain other objects as components."),
        ("Define singleton.", "A singleton is a design pattern that restricts a class to a single instance."),
        ("Define middleware.", "Middleware is software that sits between the operating system and applications."),
        ("Define API endpoint.", "An API endpoint is a URL where an API can be accessed."),
        ("Define payload.", "A payload is the data transmitted in a message or request."),
        ("Define throughput.", "Throughput is the rate at which data is successfully processed."),
        ("Define concurrency.", "Concurrency is the ability to handle multiple tasks at the same time."),
        ("Define parallelism.", "Parallelism is executing multiple tasks simultaneously."),
        ("Define deadlock.", "A deadlock occurs when two or more processes wait indefinitely for each other."),
        ("Define race condition.", "A race condition occurs when system behavior depends on timing of uncontrolled events."),
        ("Define mutex.", "A mutex is a synchronization primitive that prevents concurrent access to a resource."),
        ("Define semaphore.", "A semaphore is a variable used to control access to shared resources."),
        ("Define cache.", "A cache is temporary storage for frequently accessed data to improve performance."),
        ("Define hash.", "A hash is a fixed-size value computed from input data using a hash function."),
        ("Define token.", "A token is a small piece of data used for authentication or parsing."),
        ("Define serialization.", "Serialization converts data structures into a format for storage or transmission."),
        ("Define deserialization.", "Deserialization reconstructs data structures from stored or transmitted format."),
        ("Define refactoring.", "Refactoring is restructuring code without changing its external behavior."),
        ("Define technical debt.", "Technical debt is the cost of shortcuts taken during development."),
    ]
    for q, a in definitions:
        pairs.append((q, a, "definition"))
    
    # === WHEN/WHY QUESTIONS ===
    when_why = [
        ("When should I use Python?", "Use Python for scripting, data analysis, web backends, or when readability is a priority."),
        ("Why is encryption important?", "Encryption protects sensitive data from unauthorized access during storage and transmission."),
        ("When should I use a database?", "Use a database when you need to store, query, and manage structured data persistently."),
        ("Why use version control?", "Version control tracks changes, enables collaboration, and allows reverting to previous states."),
        ("When should I use Docker?", "Use Docker when you need consistent environments across development, testing, and production."),
        ("Why is HTTPS important?", "HTTPS encrypts data in transit, protecting it from interception and tampering."),
        ("When should I use NoSQL?", "Use NoSQL for flexible schemas, horizontal scaling, or unstructured data."),
        ("Why use cloud computing?", "Cloud computing offers scalability, reduced infrastructure costs, and global availability."),
        ("When should I refactor code?", "Refactor when code becomes hard to maintain, test, or extend."),
        ("Why is testing important?", "Testing catches bugs early, ensures code works as expected, and prevents regressions."),
        ("When should I use microservices?", "Use microservices for large, complex applications that need independent scaling."),
        ("Why use caching?", "Caching improves performance by storing frequently accessed data closer to the user."),
        ("When should I use async programming?", "Use async for I/O-bound tasks where you want to avoid blocking."),
        ("Why is logging important?", "Logging helps with debugging, monitoring, and understanding system behavior."),
        ("When should I use a load balancer?", "Use a load balancer when you need to distribute traffic across multiple servers."),
        ("Why use environment variables?", "Environment variables keep configuration separate from code and protect secrets."),
        ("When should I use indexes in a database?", "Use indexes on columns that are frequently queried or used in JOINs."),
        ("Why is input validation important?", "Input validation prevents security vulnerabilities and data corruption."),
        ("When should I use a CDN?", "Use a CDN when you need to serve static content to users globally with low latency."),
        ("Why use code reviews?", "Code reviews catch bugs, share knowledge, and maintain code quality standards."),
        ("When should I use connection pooling?", "Use connection pooling when you have many short-lived database connections."),
        ("Why is rate limiting important?", "Rate limiting prevents abuse and ensures fair resource usage."),
        ("When should I use a queue?", "Use a queue for asynchronous processing or to handle traffic spikes."),
        ("Why use dependency injection?", "Dependency injection makes code more testable and loosely coupled."),
        ("When should I use stored procedures?", "Use stored procedures for complex database logic that runs frequently."),
        ("Why is monitoring important?", "Monitoring helps detect issues early and understand system performance."),
        ("When should I use TypeScript over JavaScript?", "Use TypeScript for larger projects where type safety helps prevent bugs."),
        ("Why use linting tools?", "Linting tools enforce code style and catch common errors automatically."),
        ("When should I use an ORM?", "Use an ORM when you want to work with database records as objects."),
        ("Why is backup important?", "Backups protect against data loss from hardware failure, errors, or attacks."),
    ]
    for q, a in when_why:
        pairs.append((q, a, "when_why"))
    
    # === NUMBERS/COUNTS ===
    numbers = [
        ("How many bits in a byte?", "There are 8 bits in a byte."),
        ("How many bytes in a kilobyte?", "There are 1,024 bytes in a kilobyte."),
        ("How many HTTP status codes indicate success?", "HTTP codes in the 200-299 range indicate success."),
        ("What HTTP code means not found?", "HTTP 404 means the resource was not found."),
        ("What HTTP code means server error?", "HTTP 500 indicates an internal server error."),
        ("How many layers in the OSI model?", "The OSI model has 7 layers."),
        ("What port does HTTP use?", "HTTP typically uses port 80."),
        ("What port does HTTPS use?", "HTTPS typically uses port 443."),
        ("How many bits in an IPv4 address?", "An IPv4 address has 32 bits."),
        ("How many bits in an IPv6 address?", "An IPv6 address has 128 bits."),
        ("What port does SSH use?", "SSH typically uses port 22."),
        ("What port does FTP use?", "FTP typically uses port 21."),
        ("What port does DNS use?", "DNS typically uses port 53."),
        ("What port does MySQL use?", "MySQL typically uses port 3306."),
        ("What port does PostgreSQL use?", "PostgreSQL typically uses port 5432."),
        ("What port does Redis use?", "Redis typically uses port 6379."),
        ("What port does MongoDB use?", "MongoDB typically uses port 27017."),
        ("What HTTP code means success?", "HTTP 200 means the request was successful."),
        ("What HTTP code means created?", "HTTP 201 means a resource was created."),
        ("What HTTP code means redirect?", "HTTP 301 or 302 indicate redirects."),
        ("What HTTP code means bad request?", "HTTP 400 means the request was malformed."),
        ("What HTTP code means unauthorized?", "HTTP 401 means authentication is required."),
        ("What HTTP code means forbidden?", "HTTP 403 means access is denied."),
        ("What HTTP code means rate limited?", "HTTP 429 means too many requests."),
        ("How many megabytes in a gigabyte?", "There are 1,024 megabytes in a gigabyte."),
        ("How many gigabytes in a terabyte?", "There are 1,024 gigabytes in a terabyte."),
        ("What is the max value of a signed 32-bit integer?", "The maximum is 2,147,483,647."),
        ("What is 2 to the power of 10?", "2 to the power of 10 is 1,024."),
        ("What is 2 to the power of 8?", "2 to the power of 8 is 256."),
        ("How many colors can 8 bits represent?", "8 bits can represent 256 colors."),
    ]
    for q, a in numbers:
        pairs.append((q, a, "numbers"))
    
    # === POLITE VARIATIONS ===
    polite = [
        ("Could you explain what an API is?", "An API is an interface that allows different software systems to communicate with each other."),
        ("Would you mind telling me about Python?", "Python is a versatile, readable programming language used for many applications."),
        ("Can you help me understand encryption?", "Encryption converts data into a coded format that only authorized parties can read."),
        ("Please explain what a database is.", "A database is a structured system for storing and managing data."),
        ("I'd like to know about machine learning.", "Machine learning is a form of AI where systems learn patterns from data."),
        ("Could you tell me what Docker does?", "Docker packages applications into containers for consistent deployment across environments."),
        ("Would you explain Git to me?", "Git is a version control system that tracks changes in your code."),
        ("Can you describe cloud computing?", "Cloud computing delivers computing resources like servers and storage over the internet."),
        ("Please tell me about firewalls.", "A firewall is a security system that monitors and filters network traffic."),
        ("I want to understand what a VPN is.", "A VPN creates a secure, encrypted connection over the internet."),
        ("Could you help me with REST APIs?", "REST APIs use HTTP methods to interact with resources via URLs."),
        ("Would you mind explaining microservices?", "Microservices break applications into small, independent services."),
        ("Can you tell me about SQL?", "SQL is a language for querying and managing relational databases."),
        ("Please describe what CI/CD means.", "CI/CD automates the process of integrating, testing, and deploying code."),
        ("I'd like to understand Kubernetes.", "Kubernetes orchestrates and manages containerized applications."),
        ("Could you explain authentication to me?", "Authentication verifies who a user is through credentials."),
        ("Would you describe OAuth please?", "OAuth allows applications to access user data without sharing passwords."),
        ("Can you help me understand caching?", "Caching stores data temporarily for faster future access."),
        ("Please explain what a webhook is.", "A webhook sends automated notifications when events occur."),
        ("I want to know about load balancing.", "Load balancing distributes traffic across multiple servers."),
    ]
    for q, a in polite:
        pairs.append((q, a, "polite"))
    
    # === SECURITY FOCUSED ===
    security = [
        ("What is a SQL injection?", "SQL injection is an attack that inserts malicious SQL through user inputs."),
        ("What is XSS?", "XSS (Cross-Site Scripting) is an attack that injects malicious scripts into web pages."),
        ("What is CSRF?", "CSRF tricks users into performing unwanted actions on authenticated sites."),
        ("What is a DDoS attack?", "A DDoS attack overwhelms a server with traffic to make it unavailable."),
        ("What is phishing?", "Phishing is a social engineering attack using fake communications to steal data."),
        ("What is malware?", "Malware is malicious software designed to harm or exploit systems."),
        ("What is ransomware?", "Ransomware encrypts victim data and demands payment for decryption."),
        ("What is a zero-day vulnerability?", "A zero-day is a security flaw exploited before a fix is available."),
        ("What is a man-in-the-middle attack?", "A MITM attack intercepts communication between two parties."),
        ("What is social engineering?", "Social engineering manipulates people to reveal confidential information."),
        ("What is penetration testing?", "Penetration testing simulates attacks to find security vulnerabilities."),
        ("What is a security audit?", "A security audit is a systematic evaluation of an organization's security."),
        ("What is OWASP?", "OWASP is an organization focused on improving web application security."),
        ("What is least privilege?", "Least privilege means giving users only the access they need."),
        ("What is defense in depth?", "Defense in depth uses multiple security layers to protect systems."),
        ("What is input sanitization?", "Input sanitization cleans user input to prevent injection attacks."),
        ("What is a CVE?", "A CVE is a standardized identifier for known security vulnerabilities."),
        ("What is hashing?", "Hashing converts data into a fixed-size value that cannot be reversed."),
        ("What is salt in cryptography?", "Salt is random data added to passwords before hashing for extra security."),
        ("What is a security token?", "A security token is a device or code that proves user identity."),
    ]
    for q, a in security:
        pairs.append((q, a, "security"))
    
    # === QUICK REFERENCE ===
    quick = [
        ("Git add all files?", "Use 'git add .' to stage all changes."),
        ("Python list length?", "Use len(my_list) to get the length."),
        ("JavaScript string to number?", "Use parseInt() or Number() to convert."),
        ("Python string to int?", "Use int('123') to convert string to integer."),
        ("Current directory in bash?", "Use pwd to print working directory."),
        ("List files in Linux?", "Use ls to list files, ls -la for details."),
        ("Check Python version?", "Run python --version in terminal."),
        ("Check Node version?", "Run node --version in terminal."),
        ("Kill process in Linux?", "Use kill PID or kill -9 PID for force."),
        ("Find process in Linux?", "Use ps aux | grep process_name."),
        ("Copy file in Linux?", "Use cp source destination."),
        ("Move file in Linux?", "Use mv source destination."),
        ("Delete file in Linux?", "Use rm filename to delete."),
        ("Create directory in Linux?", "Use mkdir directory_name."),
        ("Change permissions in Linux?", "Use chmod to change file permissions."),
        ("View file contents in Linux?", "Use cat filename or less filename."),
        ("Search text in file?", "Use grep 'pattern' filename."),
        ("Count lines in file?", "Use wc -l filename."),
        ("Disk usage in Linux?", "Use df -h for disk space, du -sh for directory size."),
        ("Network info in Linux?", "Use ifconfig or ip addr to see network details."),
    ]
    for q, a in quick:
        pairs.append((q, a, "quick_ref"))
    
    return pairs


def create_episode(question: str, answer: str, episode_id: int, category: str) -> dict:
    """Create a single episode in the expected format."""
    return {
        "system": SYSTEM_PROMPT,
        "context": f"episode_id: run2_qa_{episode_id:04d}, category: {category}",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "language": "en"
    }


def main():
    output_dir = Path("data/sft_run2_minimal_qa")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mypt_run2_minimal_qa_v1.jsonl"
    
    # Generate pairs
    pairs = generate_qa_pairs()
    
    # Shuffle for variety
    random.seed(42)
    random.shuffle(pairs)
    
    # Create episodes
    episodes = []
    for i, (q, a, cat) in enumerate(pairs):
        episode = create_episode(q, a, i, cat)
        episodes.append(episode)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    
    # Stats
    total_pairs = len(pairs)
    categories = {}
    for _, _, cat in pairs:
        categories[cat] = categories.get(cat, 0) + 1
    
    avg_q_len = sum(len(q) for q, _, _ in pairs) / total_pairs
    avg_a_len = sum(len(a) for _, a, _ in pairs) / total_pairs
    
    print(f"Generated {total_pairs} Q&A pairs")
    print(f"Output: {output_file}")
    print(f"\nCategory breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print(f"\nAverage question length: {avg_q_len:.1f} chars")
    print(f"Average answer length: {avg_a_len:.1f} chars")
    print()
    print("Sample pairs:")
    for i in range(5):
        q, a, cat = pairs[i]
        print(f"  [{cat}] Q: {q}")
        print(f"         A: {a}")
        print()


if __name__ == "__main__":
    main()
