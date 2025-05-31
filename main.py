from fastapi import FastAPI, Request, HTTPException, Depends, Form, Cookie, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime, Boolean, text
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
import json
import random
import aiohttp
import asyncio
import os
import subprocess
import sys
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv
import secrets
import smtplib
from email.mime.text import MIMEText
from user_agents import parse
import platform

# Load environment variables
load_dotenv()

# Get configuration from environment variables with fallbacks
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    print("Warning: No SECRET_KEY found in environment variables. Generating a temporary one.")
    print("Please set SECRET_KEY in your .env file for production use.")
    SECRET_KEY = secrets.token_hex(32)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REMEMBER_ME_EXPIRE_DAYS = int(os.getenv("REMEMBER_ME_EXPIRE_DAYS", "30"))
RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", "15"))

# Database configuration
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./papers.db")

# Semantic Scholar API configuration
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Configuration constants
PAPER_FETCH_CONFIG = {
    "DEFAULT_TOPICS": [
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "computer vision",
        "natural language processing",
        "reinforcement learning",
        "neural networks",
        "data science",
        "robotics",
        "computer science"
    ],
    "PAPERS_PER_TOPIC": int(os.getenv("PAPERS_PER_TOPIC", "8")),
    "MIN_UNRATED_PAPERS": int(os.getenv("MIN_UNRATED_PAPERS", "10")),
    "MAX_PAPERS_PER_FETCH": int(os.getenv("MAX_PAPERS_PER_FETCH", "5")),
    "RECENT_YEARS_ONLY": os.getenv("RECENT_YEARS_ONLY", "True").lower() == "true",
    "MIN_CITATION_COUNT": int(os.getenv("MIN_CITATION_COUNT", "10"))
}

# Add session configuration after other configuration constants
SESSION_CONFIG = {
    "MAX_SESSIONS_PER_USER": int(os.getenv("MAX_SESSIONS_PER_USER", "5")),
    "INACTIVITY_TIMEOUT_MINUTES": int(os.getenv("INACTIVITY_TIMEOUT_MINUTES", "30")),
    "SESSION_LIFETIME_DAYS": int(os.getenv("SESSION_LIFETIME_DAYS", "7")),
    "CLEANUP_INTERVAL_MINUTES": int(os.getenv("CLEANUP_INTERVAL_MINUTES", "60"))
}

# Initialize password hasher
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database setup
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    
    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    helpful_papers = Column(Text, default="[]")  # JSON string of paper IDs
    not_relevant_papers = Column(Text, default="[]")  # JSON string of paper IDs
    preferred_keywords = Column(Text, default="[]")  # JSON string of keywords
    user = relationship("User", back_populates="profile")
    
    def get_helpful_papers(self) -> List[int]:
        try:
            return json.loads(self.helpful_papers)
        except:
            return []
    
    def set_helpful_papers(self, paper_ids: List[int]):
        self.helpful_papers = json.dumps(paper_ids)
    
    def get_not_relevant_papers(self) -> List[int]:
        try:
            return json.loads(self.not_relevant_papers)
        except:
            return []
    
    def set_not_relevant_papers(self, paper_ids: List[int]):
        self.not_relevant_papers = json.dumps(paper_ids)
    
    def get_preferred_keywords(self) -> List[str]:
        try:
            return json.loads(self.preferred_keywords)
        except:
            return []
    
    def set_preferred_keywords(self, keywords: List[str]):
        self.preferred_keywords = json.dumps(keywords)

class Paper(Base):
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    authors = Column(String)
    abstract = Column(Text)
    url = Column(String)
    citation_count = Column(Integer, default=0)
    year = Column(Integer)
    keywords = Column(Text, default="[]")  # JSON string of keywords, default to empty list
    reading_status = Column(String, default="unread")  # unread, reading, completed
    summary = Column(Text, default="")
    reading_notes = Column(Text, default="")
    key_takeaways = Column(Text, default="[]")  # JSON string of takeaways
    created_at = Column(String)  # ISO format timestamp
    
    def get_keywords(self) -> List[str]:
        try:
            if not self.keywords:
                return []
            return json.loads(self.keywords)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error decoding keywords JSON for paper {self.id}: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error in get_keywords for paper {self.id}: {str(e)}")
            return []
    
    def get_key_takeaways(self) -> List[str]:
        try:
            if not self.key_takeaways:
                return []
            return json.loads(self.key_takeaways)
        except json.JSONDecodeError as e:
            print(f"Error decoding key_takeaways JSON for paper {self.id}: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error in get_key_takeaways for paper {self.id}: {str(e)}")
            return []
    
    def set_key_takeaways(self, takeaways: List[str]):
        try:
            self.key_takeaways = json.dumps(takeaways)
        except Exception as e:
            print(f"Error setting key_takeaways for paper {self.id}: {str(e)}")
            self.key_takeaways = "[]"

class TokenBlacklist(Base):
    __tablename__ = "token_blacklist"
    
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token_id = Column(String, unique=True, index=True)
    device_info = Column(Text)  # JSON string containing detailed device info
    ip_address = Column(String)
    last_active = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    user = relationship("User", back_populates="sessions")
    
    def get_device_info(self) -> dict:
        try:
            return json.loads(self.device_info)
        except:
            return {}
    
    def set_device_info(self, info: dict):
        self.device_info = json.dumps(info)
    
    def is_expired(self) -> bool:
        now = datetime.utcnow()
        return (
            not self.is_active or
            self.expires_at < now or
            (now - self.last_active).total_seconds() > SESSION_CONFIG["INACTIVITY_TIMEOUT_MINUTES"] * 60
        )

# --- Pydantic Models ---
class TokenData(BaseModel):
    email: Optional[str] = None

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    token: str
    new_password: str

class UserProfileResponse(BaseModel):
    helpful_papers: List[int]
    not_relevant_papers: List[int]
    preferred_keywords: List[str]

class PaperResponse(BaseModel):
    id: int
    title: str
    authors: str
    abstract: str
    url: str
    citation_count: int
    year: int
    keywords: List[str]

# --- FastAPI App Initialization ---
app = FastAPI(title="Paper Navigator")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cookie settings based on environment
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "False").lower() == "true"  # Set to True in production

# --- Dependency Injection ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_user_profile(user: User, db: Session) -> UserProfile:
    if not user.profile:
        user_profile = UserProfile(user_id=user.id)
        db.add(user_profile)
        db.commit()
        db.refresh(user_profile)
        return user_profile
    return user.profile

# --- Email Configuration ---
SMTP_TLS = os.getenv("SMTP_TLS", "True").lower() == "true"
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")

async def send_email(to_email: str, subject: str, body: str):
    """Send an email using configured SMTP settings."""
    if not all([SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM]):
        print(f"Email not sent - SMTP not configured. Would have sent to {to_email}: {subject}")
        return False
    
    try:
        message = MIMEText(body)
        message["Subject"] = subject
        message["From"] = EMAIL_FROM
        message["To"] = to_email
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            if SMTP_TLS:
                server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

# --- Authentication Functions ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None, db: Session = None, request: Request = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Generate unique token ID
    token_id = secrets.token_hex(16)
    to_encode.update({"exp": expire, "jti": token_id})
    
    # Create session record if database and request are provided
    if db and request:
        user = db.query(User).filter(User.email == data["sub"]).first()
        if user:
            # Check session limit
            active_sessions = db.query(UserSession).filter(
                UserSession.user_id == user.id,
                UserSession.is_active == True
            ).count()
            
            if active_sessions >= SESSION_CONFIG["MAX_SESSIONS_PER_USER"]:
                # Deactivate oldest session
                oldest_session = db.query(UserSession).filter(
                    UserSession.user_id == user.id,
                    UserSession.is_active == True
                ).order_by(UserSession.last_active.asc()).first()
                
                if oldest_session:
                    oldest_session.is_active = False
                    db.commit()
            
            # Parse user agent for detailed device info
            user_agent = parse(request.headers.get("user-agent", ""))
            device_info = {
                "browser": {
                    "family": user_agent.browser.family,
                    "version": user_agent.browser.version_string
                },
                "os": {
                    "family": user_agent.os.family,
                    "version": user_agent.os.version_string
                },
                "device": {
                    "family": user_agent.device.family,
                    "brand": user_agent.device.brand,
                    "model": user_agent.device.model,
                    "is_mobile": user_agent.is_mobile,
                    "is_tablet": user_agent.is_tablet,
                    "is_pc": user_agent.is_pc
                },
                "platform": platform.platform(),
                "python_version": platform.python_version()
            }
            
            session = UserSession(
                user_id=user.id,
                token_id=token_id,
                device_info=json.dumps(device_info),
                ip_address=request.client.host if request.client else "Unknown",
                expires_at=datetime.utcnow() + timedelta(days=SESSION_CONFIG["SESSION_LIFETIME_DAYS"])
            )
            db.add(session)
            db.commit()
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_password_reset_token(email: str):
    expires = timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    return create_access_token(data={"sub": email, "type": "password_reset"}, expires_delta=expires)

def is_token_revoked(token: str, db: Session) -> bool:
    """Check if a token has been revoked."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        if not jti:
            return True
        
        # Check if token is in blacklist
        blacklisted = db.query(TokenBlacklist).filter(
            TokenBlacklist.token == jti,
            TokenBlacklist.expires_at > datetime.utcnow()
        ).first()
        return bool(blacklisted)
    except:
        return True

def revoke_token(token: str, db: Session):
    """Add a token to the blacklist."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        exp = datetime.fromtimestamp(payload.get("exp"))
        
        if jti and exp:
            blacklist_entry = TokenBlacklist(
                token=jti,
                expires_at=exp
            )
            db.add(blacklist_entry)
            db.commit()
    except:
        pass

def get_current_user(token: str = Cookie(None, alias="token"), db: Session = Depends(get_db)):
    if not token:
        return None
    try:
        # Check if token is revoked
        if is_token_revoked(token, db):
            return None
            
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        token_id: str = payload.get("jti")
        
        if email is None or token_id is None:
            return None
            
        # Get user first
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return None
            
        # Then check if session is active and not expired
        session = db.query(UserSession).filter(
            UserSession.token_id == token_id,
            UserSession.user_id == user.id
        ).first()
        
        if not session or session.is_expired():
            return None
            
        # Update last active timestamp
        session.last_active = datetime.utcnow()
        db.commit()
        
        return user
    except JWTError:
        return None

# --- HTML Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        return templates.TemplateResponse("login.html", {"request": request})
    
    user_profile = get_or_create_user_profile(current_user, db)
    total_papers = db.query(Paper).count()
    helpful_count = len(user_profile.get_helpful_papers())
    not_relevant_count = len(user_profile.get_not_relevant_papers())
    
    # Create a frequency map of preferred keywords
    keyword_counts = {}
    for keyword in user_profile.get_preferred_keywords():
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # Sort keywords by frequency, then alphabetically
    top_keywords = sorted(keyword_counts.keys(), key=lambda k: (-keyword_counts[k], k))

    return templates.TemplateResponse("home.html", {
        "request": request,
        "user": current_user,
        "total_papers": total_papers,
        "helpful_count": helpful_count,
        "not_relevant_count": not_relevant_count,
        "top_keywords": top_keywords[:10] # Show top 10
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/api/register")
async def register(user_data: UserCreate, response: Response, db: Session = Depends(get_db)):
    # Check if user already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create new user
    hashed_password = pwd_context.hash(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create access token
    access_token = create_access_token(data={"sub": new_user.email})
    
    # Set the cookie with proper attributes
    response.set_cookie(
        key="token",
        value=access_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/"  # Make cookie available for all paths
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/login")
async def login(user_data: UserLogin, request: Request, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not user.verify_password(user_data.password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # Set expiration based on remember me
    expires_delta = timedelta(days=REMEMBER_ME_EXPIRE_DAYS) if user_data.remember_me else None
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=expires_delta,
        db=db,
        request=request
    )
    
    # Set the cookie with proper attributes
    response.set_cookie(
        key="token",
        value=access_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=REMEMBER_ME_EXPIRE_DAYS * 24 * 60 * 60 if user_data.remember_me else ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/"
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/logout")
async def logout(request: Request, response: Response, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user:
        # Get the token from the request cookies
        token = request.cookies.get("token")
        if token:
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                token_id = payload.get("jti")
                if token_id:
                    # Deactivate the session
                    session = db.query(UserSession).filter(
                        UserSession.token_id == token_id,
                        UserSession.user_id == current_user.id
                    ).first()
                    if session:
                        session.is_active = False
                        db.commit()
            except JWTError:
                pass
    
    # Clear the cookie
    response.delete_cookie(
        key="token",
        path="/",
        secure=COOKIE_SECURE,
        httponly=True,
        samesite="lax"
    )
    return {"message": "Successfully logged out"}

@app.post("/api/logout-all-devices")
async def logout_all_devices(request: Request, response: Response, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user:
        # Deactivate all active sessions for the user
        db.query(UserSession).filter(
            UserSession.user_id == current_user.id,
            UserSession.is_active == True
        ).update({"is_active": False})
        db.commit()
    
    # Clear the cookie
    response.delete_cookie(
        key="token",
        path="/",
        secure=COOKIE_SECURE,
        httponly=True,
        samesite="lax"
    )
    return {"message": "Successfully logged out from all devices"}

@app.get("/api/sessions")
async def get_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Get all sessions for the user
    sessions = db.query(UserSession).filter(
        UserSession.user_id == current_user.id
    ).order_by(UserSession.last_active.desc()).all()
    
    return {
        "sessions": [
            {
                "id": session.id,
                "device_info": session.get_device_info(),
                "ip_address": session.ip_address,
                "last_active": session.last_active.isoformat(),
                "created_at": session.created_at.isoformat(),
                "is_active": session.is_active
            }
            for session in sessions
        ]
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Find and deactivate the session
    session = db.query(UserSession).filter(
        UserSession.id == session_id,
        UserSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.is_active = False
    db.commit()
    
    return {"message": "Session deactivated successfully"}

@app.get("/recommendations", response_class=HTMLResponse)
async def recommendations_page(request: Request, current_user: User = Depends(get_current_user)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("recommendations.html", {
        "request": request,
        "user": current_user
    })

@app.get("/papers", response_class=HTMLResponse)
async def all_papers(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        if not current_user:
            return RedirectResponse(url="/login", status_code=303)
        
        # Add error handling for database query
        try:
            papers = db.query(Paper).order_by(Paper.year.desc(), Paper.citation_count.desc()).all()
        except Exception as e:
            print(f"Database query error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error fetching papers from database")
        
        # Add error handling for template rendering
        try:
            return templates.TemplateResponse("all_papers.html", {
                "request": request,
                "papers": papers,
                "user": current_user
            })
        except Exception as e:
            print(f"Template rendering error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error rendering template")
            
    except Exception as e:
        print(f"Unexpected error in all_papers route: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id}", response_class=HTMLResponse)
async def paper_detail(request: Request, paper_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return templates.TemplateResponse("paper_detail.html", {
        "request": request,
        "paper": paper,
        "user": current_user
    })

# --- API Routes ---
@app.get("/api/user-profile", response_model=UserProfileResponse)
async def get_user_profile_api(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_profile = get_or_create_user_profile(current_user, db)
    return UserProfileResponse(
        helpful_papers=user_profile.get_helpful_papers(),
        not_relevant_papers=user_profile.get_not_relevant_papers(),
        preferred_keywords=user_profile.get_preferred_keywords()
    )

@app.get("/api/recommendations", response_model=PaperResponse)
async def get_recommendations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_profile = get_or_create_user_profile(current_user, db)
    rated_paper_ids = set(user_profile.get_helpful_papers() + user_profile.get_not_relevant_papers())
    
    # Fetch all unrated papers
    unrated_papers = db.query(Paper).filter(~Paper.id.in_(rated_paper_ids)).all()
    
    # If we're running low on papers, fetch more based on preferences
    if len(unrated_papers) < PAPER_FETCH_CONFIG["MIN_UNRATED_PAPERS"]:
        await fetch_papers_based_on_preferences(db, user_profile)
        # Refresh the unrated papers list
        unrated_papers = db.query(Paper).filter(~Paper.id.in_(rated_paper_ids)).all()
    
    if not unrated_papers:
        raise HTTPException(status_code=404, detail="No more papers to recommend.")
    
    # Get user's preferred keywords and their frequencies
    preferred_keywords = user_profile.get_preferred_keywords()
    keyword_frequencies = {}
    for keyword in preferred_keywords:
        keyword_frequencies[keyword] = keyword_frequencies.get(keyword, 0) + 1
    
    # Calculate paper scores based on multiple factors
    paper_scores = []
    for paper in unrated_papers:
        score = 0
        # Safely get keywords, default to empty list if None
        paper_keywords = set(paper.get_keywords() or [])
        
        # 1. Content-based scoring (keywords)
        keyword_matches = paper_keywords.intersection(set(preferred_keywords))
        if keyword_matches:
            # Weight by keyword frequency
            keyword_score = sum(keyword_frequencies.get(k, 0) for k in keyword_matches)
            score += keyword_score * 2  # Higher weight for keyword matches
        
        # 2. Recency score (recent papers get higher scores)
        current_year = datetime.now().year
        year_score = (paper.year - 2000) / (current_year - 2000)  # Normalize to 0-1
        score += year_score * 1.5
        
        # 3. Citation score (highly cited papers get higher scores)
        citation_score = min(paper.citation_count / 100, 1)  # Normalize to 0-1
        score += citation_score * 2
        
        # 4. Diversity score (papers with new keywords get a bonus)
        new_keywords = paper_keywords - set(preferred_keywords)
        if new_keywords:
            score += len(new_keywords) * 0.5
        
        paper_scores.append((paper, score))
    
    # Sort papers by score and select top 5
    top_papers = sorted(paper_scores, key=lambda x: x[1], reverse=True)[:5]
    
    if not top_papers:
        raise HTTPException(status_code=404, detail="Could not determine a recommendation.")
    
    # Randomly select from top 5 to add some variety
    selected_paper = random.choice(top_papers)[0]
    
    return PaperResponse(
        id=selected_paper.id,
        title=selected_paper.title,
        authors=selected_paper.authors,
        abstract=selected_paper.abstract or "No abstract available.",
        url=selected_paper.url or "",
        citation_count=selected_paper.citation_count or 0,
        year=selected_paper.year or datetime.now().year,
        keywords=selected_paper.get_keywords() or []
    )

@app.post("/api/rate-paper")
async def rate_paper(
    paper_id: int = Form(...),
    rating: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if rating not in ["helpful", "not_relevant"]:
        raise HTTPException(status_code=400, detail="Invalid rating. Must be 'helpful' or 'not_relevant'.")

    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    user_profile = get_or_create_user_profile(current_user, db)
    helpful_papers = user_profile.get_helpful_papers()
    not_relevant_papers = user_profile.get_not_relevant_papers()

    # Prevent re-rating
    if paper_id in helpful_papers or paper_id in not_relevant_papers:
        return {"status": "success", "message": "Paper already rated."}

    if rating == "helpful":
        helpful_papers.append(paper_id)
        user_profile.set_helpful_papers(helpful_papers)
        
        # Add paper's keywords to user's preferred list
        preferred_keywords = user_profile.get_preferred_keywords()
        preferred_keywords.extend(paper.get_keywords())
        user_profile.set_preferred_keywords(preferred_keywords)
    else: # rating == "not_relevant"
        not_relevant_papers.append(paper_id)
        user_profile.set_not_relevant_papers(not_relevant_papers)

    db.commit()
    return {"status": "success", "message": "Rating submitted successfully"}

@app.post("/api/update-reading")
async def update_reading(
    paper_id: int = Form(...),
    reading_status: str = Form(...),
    summary: str = Form(""),
    key_takeaways: str = Form("[]"),
    reading_notes: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    paper.reading_status = reading_status
    paper.summary = summary
    paper.reading_notes = reading_notes
    
    # Safely parse key_takeaways JSON
    try:
        takeaways_list = json.loads(key_takeaways)
        if isinstance(takeaways_list, list):
            paper.set_key_takeaways(takeaways_list)
        else:
            raise HTTPException(status_code=400, detail="key_takeaways must be a JSON array string.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for key_takeaways.")

    db.commit()
    return {"status": "success", "message": "Paper updated successfully."}

@app.post("/api/reset-recommendations")
async def reset_recommendations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    """Reset all user ratings and recommendations by reinitializing the database."""
    try:
        # Close current session
        db.close()
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        
        # Create all tables fresh
        Base.metadata.create_all(bind=engine)
        
        # Create new session for initialization
        new_db = SessionLocal()
        
        try:
            # Create initial user profile
            user_profile = UserProfile()
            new_db.add(user_profile)
            new_db.commit()
            
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
                },
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                    "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
                    "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
                    "url": "https://arxiv.org/abs/1810.04805",
                    "citation_count": 1200,
                    "year": 2018,
                    "keywords": ["bert", "nlp", "transformer", "language model"]
                },
                {
                    "title": "Generative Adversarial Nets",
                    "authors": "Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu",
                    "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models.",
                    "url": "https://arxiv.org/abs/1406.2661",
                    "citation_count": 1500,
                    "year": 2014,
                    "keywords": ["gan", "generative models", "adversarial learning"]
                },
                {
                    "title": "AlphaGo: Mastering the game of Go with deep neural networks and tree search",
                    "authors": "David Silver, Aja Huang, Chris J. Maddison",
                    "abstract": "We introduce a new approach to computer Go that combines deep neural networks with tree search.",
                    "url": "https://www.nature.com/articles/nature16961",
                    "citation_count": 1100,
                    "year": 2016,
                    "keywords": ["reinforcement learning", "deep learning", "game ai"]
                },
                {
                    "title": "YOLOv3: An incremental improvement",
                    "authors": "Joseph Redmon, Ali Farhadi",
                    "abstract": "We present some updates to YOLO! We made a bunch of little design changes to make it better.",
                    "url": "https://arxiv.org/abs/1804.02767",
                    "citation_count": 700,
                    "year": 2018,
                    "keywords": ["object detection", "computer vision", "real-time"]
                },
                {
                    "title": "Reinforcement Learning: An Introduction",
                    "authors": "Richard S. Sutton, Andrew G. Barto",
                    "abstract": "This book provides a clear and simple account of the key ideas and algorithms of reinforcement learning.",
                    "url": "https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf",
                    "citation_count": 2000,
                    "year": 2018,
                    "keywords": ["reinforcement learning", "machine learning", "textbook"]
                },
                {
                    "title": "ImageNet Classification with Deep Convolutional Neural Networks",
                    "authors": "Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton",
                    "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest.",
                    "url": "https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
                    "citation_count": 1800,
                    "year": 2012,
                    "keywords": ["cnn", "computer vision", "deep learning"]
                },
                {
                    "title": "A Survey of Large Language Models",
                    "authors": "Wayne Xin Zhao, Kun Zhou, Junyi Li",
                    "abstract": "This paper presents a comprehensive survey of large language models (LLMs) and their applications.",
                    "url": "https://arxiv.org/abs/2303.18223",
                    "citation_count": 300,
                    "year": 2023,
                    "keywords": ["llm", "language models", "survey"]
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
                new_db.add(paper)
            
            new_db.commit()
            
            return {
                "status": "success",
                "message": "Database successfully reset and initialized with sample papers"
            }
            
        except Exception as e:
            new_db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error during database initialization: {str(e)}"
            )
        finally:
            new_db.close()
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during database reset: {str(e)}"
        )

async def fetch_papers_from_semantic_scholar(query: str, limit: int = 10):
    """Fetch papers from Semantic Scholar API with proper error handling and rate limiting."""
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,abstract,url,year,citationCount,fieldsOfStudy"
    }
    
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
        print("Using Semantic Scholar API with key")
    else:
        print("Warning: No Semantic Scholar API key found. Using public API with limited rate.")
        # Reduce limit for public API to avoid rate limiting
        params["limit"] = min(limit, 5)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SEMANTIC_SCHOLAR_API_URL, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = data.get("data", [])
                    print(f"Successfully fetched {len(papers)} papers for query: {query}")
                    return papers
                elif response.status == 429:
                    print("Rate limit exceeded for Semantic Scholar API. Please try again later or get an API key.")
                    return []
                elif response.status == 401:
                    print("Invalid API key for Semantic Scholar. Using public API with limited rate.")
                    # Retry without API key
                    params["limit"] = min(limit, 5)
                    async with session.get(SEMANTIC_SCHOLAR_API_URL, params=params) as retry_response:
                        if retry_response.status == 200:
                            data = await retry_response.json()
                            return data.get("data", [])
                        return []
                else:
                    print(f"Error fetching papers from Semantic Scholar: {response.status}")
                    return []
    except aiohttp.ClientError as e:
        print(f"Network error while fetching papers: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error while fetching papers: {str(e)}")
        return []

async def fetch_papers_based_on_preferences(db: Session, user_profile: UserProfile):
    """Fetch new papers based on user's preferred keywords with rate limiting."""
    try:
        preferred_keywords = user_profile.get_preferred_keywords()
        if not preferred_keywords:
            # Use default topics if no preferences yet
            preferred_keywords = PAPER_FETCH_CONFIG["DEFAULT_TOPICS"][:3]  # Use first 3 default topics
        
        # Get existing paper titles to avoid duplicates
        existing_titles = {p.title for p in db.query(Paper).all()}
        
        added_papers = []
        for keyword in preferred_keywords:
            # Add a longer delay between requests when using public API
            delay = 2 if not SEMANTIC_SCHOLAR_API_KEY else 1
            await asyncio.sleep(delay)
            
            papers = await fetch_papers_from_semantic_scholar(
                keyword, 
                limit=PAPER_FETCH_CONFIG["MAX_PAPERS_PER_FETCH"]
            )
            
            for paper in papers:
                if not paper.get("title") or not paper.get("authors"):
                    continue
                
                # Skip if paper already exists
                if paper["title"] in existing_titles:
                    continue
                
                # Apply quality filters
                if PAPER_FETCH_CONFIG["RECENT_YEARS_ONLY"]:
                    current_year = datetime.now().year
                    if paper.get("year", 0) < (current_year - 5):  # Only papers from last 5 years
                        continue
                
                if paper.get("citationCount", 0) < PAPER_FETCH_CONFIG["MIN_CITATION_COUNT"]:
                    continue
                
                new_paper = Paper(
                    title=paper["title"],
                    authors=", ".join(author["name"] for author in paper.get("authors", [])),
                    abstract=paper.get("abstract", "No abstract available."),
                    url=paper.get("url", ""),
                    citation_count=paper.get("citationCount", 0),
                    year=paper.get("year", datetime.now().year),
                    keywords=json.dumps(paper.get("fieldsOfStudy", [])),
                    created_at=datetime.utcnow().isoformat()
                )
                
                db.add(new_paper)
                added_papers.append(new_paper)
                existing_titles.add(paper["title"])
        
        if added_papers:
            db.commit()
            print(f"Added {len(added_papers)} new papers")
        
        return len(added_papers)
    except Exception as e:
        db.rollback()
        print(f"Error fetching papers based on preferences: {str(e)}")
        return 0

@app.post("/api/fetch-new-papers")
async def fetch_new_papers(
    query: str = Form(...),
    limit: int = Form(10),
    db: Session = Depends(get_db)
):
    """Fetch new papers from Semantic Scholar and add them to the database."""
    try:
        # Fetch papers from Semantic Scholar
        papers = await fetch_papers_from_semantic_scholar(query, limit)
        
        added_papers = []
        for paper in papers:
            # Skip papers with missing required fields
            if not paper.get("title") or not paper.get("authors"):
                continue
                
            # Check if paper already exists
            existing_paper = db.query(Paper).filter(Paper.title == paper["title"]).first()
            if existing_paper:
                continue
            
            # Create new paper with safe defaults for missing fields
            new_paper = Paper(
                title=paper["title"],
                authors=", ".join(author["name"] for author in paper.get("authors", [])),
                abstract=paper.get("abstract", "No abstract available."),
                url=paper.get("url", ""),
                citation_count=paper.get("citationCount", 0),
                year=paper.get("year", datetime.now().year),
                keywords=json.dumps(paper.get("fieldsOfStudy", []))
            )
            
            db.add(new_paper)
            added_papers.append(new_paper)
        
        db.commit()
        
        return {
            "status": "success",
            "message": f"Added {len(added_papers)} new papers",
            "papers": [
                {
                    "id": paper.id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year
                }
                for paper in added_papers
            ]
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/request-password-reset")
async def request_password_reset(request: PasswordResetRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        # Don't reveal that the email doesn't exist
        return {"message": "If your email is registered, you will receive a password reset link"}
    
    reset_token = create_password_reset_token(request.email)
    reset_link = f"{os.getenv('BASE_URL', 'http://localhost:8000')}/reset-password?token={reset_token}"
    
    # Send reset link via email
    email_sent = await send_email(
        to_email=request.email,
        subject="Password Reset Request",
        body=f"""Hello,

You have requested to reset your password. Click the link below to set a new password:

{reset_link}

This link will expire in {RESET_TOKEN_EXPIRE_MINUTES} minutes.

If you did not request this password reset, please ignore this email.

Best regards,
Paper Navigator Team"""
    )
    
    if not email_sent:
        raise HTTPException(
            status_code=500,
            detail="Failed to send password reset email. Please try again later."
        )
    
    return {"message": "If your email is registered, you will receive a password reset link"}

@app.post("/api/reset-password")
async def reset_password(reset_data: PasswordReset, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(reset_data.token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        token_type = payload.get("type")
        
        if not email or token_type != "password_reset":
            raise HTTPException(status_code=400, detail="Invalid reset token")
        
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update password
        user.hashed_password = pwd_context.hash(reset_data.new_password)
        db.commit()
        
        return {"message": "Password successfully reset"}
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

# Add new routes for password reset pages
@app.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str = None):
    return templates.TemplateResponse("reset_password.html", {
        "request": request,
        "token": token
    })

@app.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {
        "request": request
    })

# --- Application Startup ---
@app.on_event("startup")
async def startup_event():
    try:
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        
        # Create a new session for initialization
        db = SessionLocal()
        try:
            # Check if we need to add the expires_at column
            try:
                # Try to query the expires_at column
                db.execute(text("SELECT expires_at FROM user_sessions LIMIT 1"))
            except Exception:
                # If the column doesn't exist, add it
                print("Adding expires_at column to user_sessions table...")
                db.execute(text("ALTER TABLE user_sessions ADD COLUMN expires_at DATETIME"))
                db.commit()
            
            # Clean up expired sessions
            db.query(UserSession).filter(
                UserSession.expires_at < datetime.utcnow()
            ).delete()
            db.commit()
            
            # Update any papers with None or invalid keywords
            papers = db.query(Paper).filter(
                (Paper.keywords == None) | 
                (Paper.keywords == "") | 
                (Paper.keywords == "null")
            ).all()
            
            for paper in papers:
                paper.keywords = "[]"
            
            if papers:
                db.commit()
                print(f"Updated {len(papers)} papers with invalid keywords")
            
            # Check if we have any papers
            paper_count = db.query(Paper).count()
            
            # Only add sample papers if the database is empty
            if paper_count == 0:
                print("Adding initial sample papers...")
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
                print(f"Added {len(initial_papers)} sample papers")
            else:
                print(f"Database already contains {paper_count} papers")
            
        except Exception as e:
            db.rollback()
            print(f"Error during database initialization: {str(e)}")
            raise
        finally:
            db.close()
            
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise

# Add a periodic cleanup task
async def cleanup_expired_sessions():
    while True:
        try:
            db = SessionLocal()
            try:
                # Clean up expired sessions
                db.query(UserSession).filter(
                    UserSession.expires_at < datetime.utcnow()
                ).delete()
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"Error during session cleanup: {str(e)}")
            finally:
                db.close()
        except Exception as e:
            print(f"Error during session cleanup: {str(e)}")
        
        await asyncio.sleep(SESSION_CONFIG["CLEANUP_INTERVAL_MINUTES"] * 60)

# Start the cleanup task
@app.on_event("startup")
async def start_cleanup_task():
    asyncio.create_task(cleanup_expired_sessions())