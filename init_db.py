from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from main import Base, User, UserProfile, Paper
from passlib.context import CryptContext
import json
from datetime import datetime
import os

# Initialize password hasher
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create database directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Create SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///data/papers.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(String)  # ISO format timestamp
    preferences = Column(Text, default="{}")  # JSON string of user preferences
    
    def get_preferences(self) -> dict:
        try:
            return json.loads(self.preferences)
        except:
            return {}
    
    def set_preferences(self, prefs: dict):
        self.preferences = json.dumps(prefs)

class Paper(Base):
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    authors = Column(String)
    abstract = Column(Text)
    url = Column(String)
    pdf_url = Column(String)
    published_date = Column(String)
    added_date = Column(String)  # ISO format timestamp
    reading_status = Column(String, default="unread")  # unread, reading, completed
    user_notes = Column(Text, default="")
    key_takeaways = Column(Text, default="")
    citation_count = Column(Integer, default=0)
    user_id = Column(Integer, ForeignKey("users.id"))

class PaperCollection(Base):
    __tablename__ = "paper_collections"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(String)  # ISO format timestamp
    papers = Column(Text, default="[]")  # JSON string of paper IDs
    
    def get_papers(self) -> list:
        try:
            return json.loads(self.papers)
        except:
            return []
    
    def set_papers(self, paper_ids: list):
        self.papers = json.dumps(paper_ids)

def init_db():
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create a new session
    db = SessionLocal()
    
    try:
        # Create default admin user
        admin_user = User(
            email="admin@example.com",
            username="admin",
            hashed_password=pwd_context.hash("admin123"),  # Change this in production!
            created_at=datetime.utcnow().isoformat()
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        # Create user profile for admin
        admin_profile = UserProfile(
            user_id=admin_user.id,
            helpful_papers="[]",
            not_relevant_papers="[]",
            preferred_keywords=json.dumps(["machine learning", "artificial intelligence", "deep learning"])
        )
        db.add(admin_profile)
        
        # Sample papers for initial database
        initial_papers = [
            {
                "title": "Deep Learning",
                "authors": "Yann LeCun, Yoshua Bengio, Geoffrey Hinton",
                "abstract": "Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction.",
                "url": "https://www.nature.com/articles/nature14539",
                "citation_count": 1000,
                "year": 2015,
                "keywords": ["deep learning", "neural networks", "machine learning"]
            },
            {
                "title": "Attention Is All You Need",
                "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar",
                "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                "url": "https://arxiv.org/abs/1706.03762",
                "citation_count": 800,
                "year": 2017,
                "keywords": ["transformer", "attention", "natural language processing"]
            },
            {
                "title": "ResNet: Deep Residual Learning for Image Recognition",
                "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren",
                "abstract": "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.",
                "url": "https://arxiv.org/abs/1512.03385",
                "citation_count": 900,
                "year": 2015,
                "keywords": ["computer vision", "deep learning", "residual networks"]
            }
        ]
        
        # Add initial papers
        for paper_data in initial_papers:
            paper = Paper(
                title=paper_data["title"],
                authors=paper_data["authors"],
                abstract=paper_data["abstract"],
                url=paper_data["url"],
                citation_count=paper_data["citation_count"],
                year=paper_data["year"],
                keywords=json.dumps(paper_data["keywords"]),
                created_at=datetime.utcnow().isoformat()
            )
            db.add(paper)
        
        db.commit()
        print("Database initialized successfully!")
        print("Default admin user created:")
        print("Email: admin@example.com")
        print("Password: admin123")
        print("Please change these credentials in production!")
        
    except Exception as e:
        db.rollback()
        print(f"Error initializing database: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()