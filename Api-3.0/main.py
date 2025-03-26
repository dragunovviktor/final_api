from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, validator,AnyUrl, Field
from typing import List, Optional
from datetime import datetime
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uuid
from typing import List, Optional, Union


DATABASE_URL = "postgresql://sber_admin:securepass@db:5432/sber_api"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Модели базы данных
class Branch(Base):
    __tablename__ = "branches"
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, index=True)
    internal_code = Column(String, unique=True)
    latitude = Column(String)
    longitude = Column(String)


class ObjectType(Base):
    __tablename__ = "object_types"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)  # газон, крыльцо, тротуар
    measure_unit = Column(String)  # м², пог. м


class BranchObject(Base):
    __tablename__ = "branch_objects"
    id = Column(Integer, primary_key=True, index=True)
    branch_id = Column(Integer, ForeignKey("branches.id"))
    object_type_id = Column(Integer, ForeignKey("object_types.id"))
    name = Column(String)
    area = Column(String)
    description = Column(String)


class MaintenancePlan(Base):
    __tablename__ = "maintenance_plans"
    id = Column(Integer, primary_key=True, index=True)
    branch_id = Column(Integer, ForeignKey("branches.id"))
    object_id = Column(Integer, ForeignKey("branch_objects.id"), nullable=True)
    work_type = Column(String)
    frequency = Column(String)
    next_maintenance_date = Column(DateTime)


class CompletedWork(Base):
    __tablename__ = "completed_works"
    id = Column(Integer, primary_key=True, index=True)
    branch_id = Column(Integer, ForeignKey("branches.id"))
    object_id = Column(Integer, ForeignKey("branch_objects.id"), nullable=True)
    work_type = Column(String)
    completion_date = Column(DateTime)
    responsible_person = Column(String)
    notes = Column(Text)


class BranchAttachment(Base):
    __tablename__ = "branch_attachments"
    id = Column(Integer, primary_key=True, index=True)
    branch_id = Column(Integer, ForeignKey("branches.id"))
    object_id = Column(Integer, ForeignKey("branch_objects.id"), nullable=True)
    file_type = Column(String)  # фото, схема, план
    file_url = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Sber Branches API",
    description="API для управления территориями отделений Сбербанка",
    version="1.0.0",
    docs_url="/docs",  # Измените на /docs вместо /api/docs
    redoc_url="/redoc"  # Измените на /redoc вместо /api/redoc
)

# Pydantic модели

class BranchBase(BaseModel):
    address: str
    internal_code: str


class BranchCreate(BaseModel):
    address: str
    internal_code: str
    latitude: Optional[Union[float, str]] = None  # Принимаем и float, и string
    longitude: Optional[Union[float, str]] = None

class BranchCreate(BranchBase):
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @validator('latitude', 'longitude')
    def validate_coordinates(cls, v):
        if v is not None:
            try:
                float(v)
            except ValueError:
                raise ValueError("Координата должна быть числом")
        return v

class BranchResponse(BranchBase):
    latitude: Optional[str] = None
    longitude: Optional[str] = None

    class Config:
        orm_mode = True


class AttachmentResponse(BaseModel):
    id: int
    branch_id: int  # Храним ID, но в ответ добавляем код
    branch_code: str = Field(..., alias="branch.internal_code")  # Получаем код из связи
    object_id: Optional[int]
    file_type: str
    file_url: str
    uploaded_at: datetime

    class Config:
        orm_mode = True
        allow_population_by_field_name = True

class AttachmentCreate(BaseModel):
    branch_code: str
    object_id: Optional[int] = None
    file_type: str = "photo"
    file_url: AnyUrl  # Автоматическая валидация URL


class ObjectCreate(BaseModel):
    branch_code: str
    object_type_id: int
    name: str
    area: str
    description: Optional[str] = None

class MaintenancePlanCreate(BaseModel):
    branch_code: str
    object_id: Optional[int] = None
    work_type: str
    frequency: str
    next_maintenance_date: datetime


class CompletedWorkCreate(BaseModel):
    branch_code: str
    object_id: Optional[int] = None
    work_type: str
    completion_date: datetime
    responsible_person: str
    notes: Optional[str] = None

class ObjectTypeCreate(BaseModel):
    name: str
    measure_unit: str

class ObjectTypeResponse(ObjectTypeCreate):
    id: int
    
    class Config:
        orm_mode = True

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Эндпоинты для администраторов
@app.post("/api/branches", response_model=BranchResponse)
def create_branch(branch: BranchCreate, db: Session = Depends(get_db)):
    existing = db.query(Branch).filter(Branch.internal_code == branch.internal_code).first()
    if existing:
        raise HTTPException(status_code=400, detail="Филиал с таким кодом уже существует")

    db_branch = Branch(
        address=branch.address,
        internal_code=branch.internal_code,
        latitude=str(branch.latitude) if branch.latitude is not None else None,
        longitude=str(branch.longitude) if branch.longitude is not None else None
    )

    db.add(db_branch)
    db.commit()
    db.refresh(db_branch)
    return db_branch

@app.get("/api/branches/by-code/{branch_code}/attachments", response_model=List[AttachmentResponse])
def get_branch_attachments(
    branch_code: str,
    object_id: Optional[int] = None,
    file_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        # Находим филиал с JOIN к attachments
        branch = db.query(Branch)\
            .filter(Branch.internal_code == branch_code)\
            .first()
        
        if not branch:
            raise HTTPException(status_code=404, detail="Филиал не найден")

        # Строим запрос с явным указанием полей
        query = db.query(
            BranchAttachment.id,
            BranchAttachment.branch_id,
            Branch.internal_code.label("branch_code"),
            BranchAttachment.object_id,
            BranchAttachment.file_type,
            BranchAttachment.file_url,
            BranchAttachment.uploaded_at
        ).join(Branch)\
         .filter(BranchAttachment.branch_id == branch.id)

        # Применяем фильтры
        if object_id is not None:
            query = query.filter(BranchAttachment.object_id == object_id)
        
        if file_type is not None:
            query = query.filter(BranchAttachment.file_type == file_type)

        # Выполняем запрос и преобразуем результат
        attachments = query.all()
        
        return [{
            "id": a.id,
            "branch_id": a.branch_id,
            "branch_code": a.branch_code,
            "object_id": a.object_id,
            "file_type": a.file_type,
            "file_url": a.file_url,
            "uploaded_at": a.uploaded_at
        } for a in attachments]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении вложений: {str(e)}"
        )

@app.put("/api/branches/by-code/{internal_code}", response_model=BranchResponse)
def update_branch_by_code(internal_code: str, branch: BranchBase, db: Session = Depends(get_db)):
    # Находим филиал по internal_code
    db_branch = db.query(Branch).filter(Branch.internal_code == internal_code).first()
    if not db_branch:

        raise HTTPException(status_code=404, detail="Филиал с указанным кодом не найден")

    # Обновляем только переданные поля (исключая internal_code)
    update_data = branch.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_branch, field, value)

    db.commit()
    db.refresh(db_branch)
    return db_branch

@app.post("/api/object-types", response_model=ObjectTypeResponse)
def create_object_type(obj_type: ObjectTypeCreate, db: Session = Depends(get_db)):
    # Проверяем, нет ли уже типа с таким именем
    existing = db.query(ObjectType).filter(ObjectType.name == obj_type.name).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Тип объекта с названием '{obj_type.name}' уже существует"
        )
    
    db_type = ObjectType(
        name=obj_type.name,
        measure_unit=obj_type.measure_unit
    )
    db.add(db_type)
    db.commit()
    db.refresh(db_type)
    return db_type


@app.put("/api/branches/{branch_id}", response_model=BranchResponse)
def update_branch(branch_id: int, branch: BranchCreate, db: Session = Depends(get_db)):
    db_branch = db.query(Branch).filter(Branch.id == branch_id).first()
    if not db_branch:
        raise HTTPException(status_code=404, detail="Филиал не найден")

    update_data = branch.dict(exclude_unset=True)
    for field, value in update_data.items():
        if field in ['latitude', 'longitude'] and value is not None:
            setattr(db_branch, field, str(value))
        elif value is not None:
            setattr(db_branch, field, value)

    db.commit()
    db.refresh(db_branch)
    return db_branch


@app.post("/api/objects", response_model=ObjectCreate)
def create_object(obj: ObjectCreate, db: Session = Depends(get_db)):
    # Находим филиал по internal_code
    branch = db.query(Branch).filter(Branch.internal_code == obj.branch_code).first()
    if not branch:
        raise HTTPException(status_code=404, detail="Филиал с указанным кодом не найден")

    # Проверяем существование типа объекта
    if not db.query(ObjectType).filter(ObjectType.id == obj.object_type_id).first():
        raise HTTPException(status_code=404, detail="Тип объекта не найден")

    db_object = BranchObject(
        branch_id=branch.id,  # Используем найденный ID филиала
        object_type_id=obj.object_type_id,
        name=obj.name,
        area=obj.area,
        description=obj.description
    )

    db.add(db_object)
    db.commit()
    db.refresh(db_object)

    return {
        "branch_code": obj.branch_code,  # Возвращаем переданный код
        "object_type_id": db_object.object_type_id,
        "name": db_object.name,
        "area": db_object.area,
        "description": db_object.description
    }


@app.post("/api/maintenance", response_model=MaintenancePlanCreate)
def create_maintenance_plan(plan: MaintenancePlanCreate, db: Session = Depends(get_db)):
    # Находим филиал по коду
    branch = db.query(Branch).filter(Branch.internal_code == plan.branch_code).first()
    if not branch:
        raise HTTPException(status_code=404, detail="Филиал с указанным кодом не найден")

    # Проверяем объект, если указан
    if plan.object_id:
        if not db.query(BranchObject).filter(BranchObject.id == plan.object_id).first():
            raise HTTPException(status_code=404, detail="Объект филиала не найден")

    db_plan = MaintenancePlan(
        branch_id=branch.id,  # Используем ID найденного филиала
        object_id=plan.object_id,
        work_type=plan.work_type,
        frequency=plan.frequency,
        next_maintenance_date=plan.next_maintenance_date
    )

    db.add(db_plan)
    db.commit()
    db.refresh(db_plan)
    
    # Возвращаем данные с подставленным branch_code
    return {
        "branch_code": plan.branch_code,
        "object_id": db_plan.object_id,
        "work_type": db_plan.work_type,
        "frequency": db_plan.frequency,
        "next_maintenance_date": db_plan.next_maintenance_date
    }

@app.post("/api/completed-works", response_model=CompletedWorkCreate)
def create_completed_work(work: CompletedWorkCreate, db: Session = Depends(get_db)):
    # Находим филиал по коду
    branch = db.query(Branch).filter(Branch.internal_code == work.branch_code).first()
    if not branch:
        raise HTTPException(status_code=404, detail="Филиал с указанным кодом не найден")

    # Проверяем объект, если указан
    if work.object_id:
        if not db.query(BranchObject).filter(BranchObject.id == work.object_id).first():
            raise HTTPException(status_code=404, detail="Объект филиала не найден")

    db_work = CompletedWork(
        branch_id=branch.id,  # Используем ID найденного филиала
        object_id=work.object_id,
        work_type=work.work_type,
        completion_date=work.completion_date,
        responsible_person=work.responsible_person,
        notes=work.notes
    )

    db.add(db_work)
    db.commit()
    db.refresh(db_work)
    
    # Возвращаем данные с подставленным branch_code
    return {
        "branch_code": work.branch_code,
        "object_id": db_work.object_id,
        "work_type": db_work.work_type,
        "completion_date": db_work.completion_date,
        "responsible_person": db_work.responsible_person,
        "notes": db_work.notes
    }

@app.post("/api/attachments", response_model=AttachmentCreate)
async def create_attachment(
    attachment: AttachmentCreate,
    db: Session = Depends(get_db)
):
    # 1. Находим филиал по коду
    branch = db.query(Branch).filter(Branch.internal_code == attachment.branch_code).first()
    if not branch:
        raise HTTPException(status_code=404, detail="Филиал не найден")

    # 2. Проверяем объект, если указан
    if attachment.object_id:
        obj = db.query(BranchObject).filter(
            BranchObject.id == attachment.object_id,
            BranchObject.branch_id == branch.id
        ).first()
        if not obj:
            raise HTTPException(status_code=404, detail="Объект не найден")

    # 3. Сохраняем в базу
    db_attachment = BranchAttachment(
        branch_id=branch.id,
        object_id=attachment.object_id,
        file_type=attachment.file_type,
        file_url=str(attachment.file_url)  # Преобразуем HttpUrl в строку
    )
    
    db.add(db_attachment)
    db.commit()
    db.refresh(db_attachment)

    return {
        "branch_code": attachment.branch_code,
        "object_id": attachment.object_id,
        "file_type": attachment.file_type,
        "file_url": attachment.file_url
    }

# Эндпоинты для чат-бота
@app.get("/api/branches", response_model=List[BranchCreate])
def search_branches(search: str = "", db: Session = Depends(get_db)):
    branches = db.query(Branch).filter(
        (Branch.address.ilike(f"%{search}%")) |
        (Branch.internal_code.ilike(f"%{search}%"))
    ).all()

    # Преобразуем SQLAlchemy объекты в словари
    return [
        {
            "address": branch.address,
            "internal_code": branch.internal_code,
            "latitude": branch.latitude,
            "longitude": branch.longitude
        }
        for branch in branches
    ]

@app.get("/api/branches/by-code/{branch_code}/objects", response_model=List[ObjectCreate])
def get_branch_objects_by_code(branch_code: str, db: Session = Depends(get_db)):
    branch = db.query(Branch).filter(Branch.internal_code == branch_code).first()
    if not branch:
        raise HTTPException(status_code=404, detail="Филиал не найден")

    objects = db.query(BranchObject).filter(BranchObject.branch_id == branch.id).all()
    return [
        {
            "branch_code": branch_code,
            "object_type_id": obj.object_type_id,
            "name": obj.name,
            "area": obj.area,
            "description": obj.description
        }
        for obj in objects
    ]


@app.get("/api/branches/by-code/{branch_code}/plans", response_model=List[MaintenancePlanCreate])
def get_branch_plans_by_code(branch_code: str, db: Session = Depends(get_db)):
    # Находим филиал по коду
    branch = db.query(Branch).filter(Branch.internal_code == branch_code).first()
    if not branch:
        raise HTTPException(status_code=404, detail="Филиал не найден")

    # Получаем планы для найденного филиала
    plans = db.query(MaintenancePlan).filter(MaintenancePlan.branch_id == branch.id).all()
    
    return [
        {
            "branch_code": branch_code,  # Возвращаем код вместо ID
            "object_id": plan.object_id,
            "work_type": plan.work_type,
            "frequency": plan.frequency,
            "next_maintenance_date": plan.next_maintenance_date
        }
        for plan in plans
    ]

@app.get("/api/branches/by-code/{branch_code}/completed-works", response_model=List[CompletedWorkCreate])
def get_branch_completed_works_by_code(branch_code: str, db: Session = Depends(get_db)):
    # Находим филиал по коду
    branch = db.query(Branch).filter(Branch.internal_code == branch_code).first()
    if not branch:
        raise HTTPException(status_code=404, detail="Филиал не найден")

    # Получаем выполненные работы для филиала
    works = db.query(CompletedWork).filter(CompletedWork.branch_id == branch.id).all()
    
    return [
        {
            "branch_code": branch_code,  # Возвращаем код вместо ID
            "object_id": work.object_id,
            "work_type": work.work_type,
            "completion_date": work.completion_date,
            "responsible_person": work.responsible_person,
            "notes": work.notes
        }
        for work in works
    ]

@app.get("/api/branches/by-code/{branch_code}/attachments", response_model=List[dict])
def get_branch_attachments_by_code(branch_code: str, db: Session = Depends(get_db)):
    # Сначала находим филиал по его коду
    branch = db.query(Branch).filter(Branch.internal_code == branch_code).first()
    if not branch:
        raise HTTPException(status_code=404, detail="Филиал с указанным кодом не найден")
    
    # Затем получаем все вложения для этого филиала
    attachments = db.query(BranchAttachment).filter(BranchAttachment.branch_id == branch.id).all()
    
    return [
        {
            "id": attachment.id,
            "branch_code": branch_code,  # Возвращаем код вместо ID
            "object_id": attachment.object_id,
            "file_type": attachment.file_type,
            "file_url": attachment.file_url,
            "uploaded_at": attachment.uploaded_at
        }
        for attachment in attachments
    ]

@app.get("/api/nlp-query")
def process_nlp_query(query: str, db: Session = Depends(get_db)):
    if "планируется" in query.lower() or "планы" in query.lower():
        branch_code = "".join([c for c in query if c.isdigit()])
        if branch_code:
            try:
                plans = get_branch_plans(int(branch_code), db)
                return {
                    "status": "success",
                    "data": plans,
                    "message": f"Найдены планы для филиала {branch_code}"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Ошибка при получении планов: {str(e)}"
                }

    elif "выполнено" in query.lower() or "сделано" in query.lower():
        branch_code = "".join([c for c in query if c.isdigit()])
        if branch_code:
            try:
                works = get_branch_completed_works(int(branch_code), db)
                return {
                    "status": "success",
                    "data": works,
                    "message": f"Найдены выполненные работы для филиала {branch_code}"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Ошибка при получении выполненных работ: {str(e)}"
                }

    return {
        "status": "not_found",
        "message": "Не удалось обработать запрос. Уточните параметры поиска."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9080)
