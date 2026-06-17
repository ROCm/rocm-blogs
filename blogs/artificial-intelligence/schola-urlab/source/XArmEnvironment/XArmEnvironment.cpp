// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Copy into your game module's Private/ folder.

#include "XArmEnvironment.h"

#include "Common/InteractionDefinition.h"
#include "GameFramework/Actor.h"
#include "MuJoCo/Components/Actuators/MjActuator.h"
#include "MuJoCo/Components/Controllers/MjArticulationController.h"
#include "MuJoCo/Components/Joints/MjJoint.h"
#include "MuJoCo/Components/Sensors/MjSensor.h"
#include "MuJoCo/Core/AMjManager.h"
#include "MuJoCo/Core/MjArticulation.h"
#include "MuJoCo/Core/MjPhysicsEngine.h"
#include "Points/BoxPoint.h"
#include "Points/DictPoint.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DictSpace.h"
#include "StructUtils/InstancedStruct.h"

DEFINE_LOG_CATEGORY_STATIC(LogXArmEnv, Log, All);

namespace
{
	const TArray<FString> ReachObservationSensorNames = { TEXT("tcp_pos") };

	const TArray<FString> ReachObservationJointNames = {
		TEXT("joint1"),
		TEXT("joint2"),
		TEXT("joint3"),
		TEXT("joint4"),
		TEXT("joint5"),
		TEXT("joint6"),
	};

	const TArray<FString> ReachActionActuatorNames = {
		TEXT("joint1_act"),
		TEXT("joint2_act"),
		TEXT("joint3_act"),
		TEXT("joint4_act"),
		TEXT("joint5_act"),
		TEXT("joint6_act"),
	};

	const FString ReachResetKeyframeName = TEXT("home");

	FVector LocalMetersToUeCm(const FVector& Meters)
	{
		return FVector(Meters.X * 100.0, Meters.Y * -100.0, Meters.Z * 100.0);
	}

	struct FReachMetrics
	{
		FVector TcpWorldCm = FVector::ZeroVector;
		float DistanceMeters = 0.f;
		float JointVelocitySquaredSum = 0.f;
	};

	FBoxSpace MakeBoxSpace(int32 Dim, float Low, float High)
	{
		FBoxSpace Space;
		for (int32 i = 0; i < FMath::Max(Dim, 1); ++i)
		{
			Space.Add(Low, High);
		}
		return Space;
	}

	void AddBoxSubSpace(FDictSpace& DictSpace, const FString& Key, int32 Dim, float Low, float High)
	{
		DictSpace.Spaces.Add(Key, TInstancedStruct<FSpace>::Make<FBoxSpace>(MakeBoxSpace(Dim, Low, High)));
	}

	void SetBoxSpaceBounds(FDictSpace& DictSpace, float Low, float High)
	{
		for (TPair<FString, TInstancedStruct<FSpace>>& Pair : DictSpace.Spaces)
		{
			if (FBoxSpace* Box = Pair.Value.GetMutablePtr<FBoxSpace>())
			{
				for (FBoxSpaceDimension& Dimension : Box->Dimensions)
				{
					Dimension.Low = Low;
					Dimension.High = High;
				}
			}
		}
	}

	float ScaleAndClip(float Value, float Scale, float Clip)
	{
		return FMath::Clamp(Value / FMath::Max(Scale, KINDA_SMALL_NUMBER), -Clip, Clip);
	}

	void SetScaledVector(TArray<float>& Values, const FVector& Vector, float Scale, float Clip)
	{
		if (Values.Num() < 3)
		{
			return;
		}

		Values[0] = ScaleAndClip(static_cast<float>(Vector.X), Scale, Clip);
		Values[1] = ScaleAndClip(static_cast<float>(Vector.Y), Scale, Clip);
		Values[2] = ScaleAndClip(static_cast<float>(Vector.Z), Scale, Clip);
	}

	bool BuildReachMetrics(const FURLabScholaAgentSnapshot& Snapshot, const FString& TcpSensorName,
		const FVector& TargetWorldCm, FReachMetrics& OutMetrics)
	{
		OutMetrics = FReachMetrics{};
		bool bFoundTcp = false;

		for (const FURLabScholaNamedSensorReading& Row : Snapshot.Observations)
		{
			if (Row.SensorName == TcpSensorName && Row.Values.Num() >= 3)
			{
				OutMetrics.TcpWorldCm = FVector(Row.Values[0], Row.Values[1], Row.Values[2]);
				bFoundTcp = true;
				break;
			}
		}

		if (!bFoundTcp)
		{
			return false;
		}

		OutMetrics.DistanceMeters = FVector::Dist(OutMetrics.TcpWorldCm, TargetWorldCm) / 100.f;
		for (const FString& JointName : ReachObservationJointNames)
		{
			for (const FURLabScholaNamedSensorReading& Row : Snapshot.Observations)
			{
				if (Row.SensorName == JointName && Row.Values.Num() >= 2)
				{
					const float Vel = Row.Values[1];
					OutMetrics.JointVelocitySquaredSum += Vel * Vel;
					break;
				}
			}
		}
		return true;
	}
}

AXArmEnvironment::AXArmEnvironment()
{
	PrimaryActorTick.bCanEverTick = false;
	EpisodeTargetLocationMeters = TargetLocationMeters;
}

void AXArmEnvironment::BeginPlay()
{
	Super::BeginPlay();
	ApplyControlSourceIfNeeded();
	SampleEpisodeTargetLocation();
	SyncTargetActorToLocation();
}

void AXArmEnvironment::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);
	EpisodeTargetLocationMeters = TargetLocationMeters;
	UpdateTargetWorldCm();
	SyncTargetActorToLocation();
}

#if WITH_EDITOR
void AXArmEnvironment::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);

	const FName PropName = PropertyChangedEvent.GetPropertyName();
	static const FName NAME_TargetLocationMeters = GET_MEMBER_NAME_CHECKED(AXArmEnvironment, TargetLocationMeters);
	static const FName NAME_TargetRandomOffsetMaxMeters = GET_MEMBER_NAME_CHECKED(AXArmEnvironment, TargetRandomOffsetMaxMeters);
	static const FName NAME_bRandomizeTargetPerEpisode = GET_MEMBER_NAME_CHECKED(AXArmEnvironment, bRandomizeTargetPerEpisode);
	static const FName NAME_TargetActor = GET_MEMBER_NAME_CHECKED(AXArmEnvironment, TargetActor);

	if (PropName == NAME_TargetLocationMeters
		|| PropName == NAME_TargetRandomOffsetMaxMeters
		|| PropName == NAME_bRandomizeTargetPerEpisode
		|| PropName == NAME_TargetActor)
	{
		EpisodeTargetLocationMeters = TargetLocationMeters;
		UpdateTargetWorldCm();
		SyncTargetActorToLocation();
	}
}
#endif

void AXArmEnvironment::ApplyControlSourceIfNeeded()
{
	if (!RobotArticulation)
	{
		return;
	}

	if (bForceUiControlSource)
	{
		RobotArticulation->ControlSource = 1;
	}

	TArray<UMjArticulationController*> Controllers;
	RobotArticulation->GetComponents<UMjArticulationController>(Controllers);
	for (UMjArticulationController* Controller : Controllers)
	{
		if (Controller && Controller->bEnabled)
		{
			Controller->bEnabled = false;
			UE_LOG(LogXArmEnv, Log,
				TEXT("Disabled articulation controller '%s' on '%s' so policy actions reach actuators."),
				*Controller->GetName(), *RobotArticulation->GetName());
		}
	}
}

void AXArmEnvironment::RebuildComponentCaches()
{
	CachedSensors.Empty();
	CachedJoints.Empty();
	CachedActuators.Empty();

	if (!RobotArticulation)
	{
		return;
	}

	auto CacheComponents = [this](const TArray<FString>& Names, auto& Cache, const TCHAR* MissingLabel, auto Lookup)
	{
		for (const FString& Name : Names)
		{
			if (auto* Component = Lookup(Name))
			{
				Cache.Add(Name, Component);
			}
			else
			{
				UE_LOG(LogXArmEnv, Warning, TEXT("%s '%s' not found on articulation '%s'."),
					MissingLabel, *Name, *RobotArticulation->GetName());
			}
		}
	};

	CacheComponents(ReachObservationSensorNames, CachedSensors, TEXT("Observation sensor"),
		[this](const FString& Name) { return RobotArticulation->GetSensor(Name); });
	CacheComponents(ReachObservationJointNames, CachedJoints, TEXT("Observation joint"),
		[this](const FString& Name) { return RobotArticulation->GetJoint(Name); });
	CacheComponents(ReachActionActuatorNames, CachedActuators, TEXT("Actuator"),
		[this](const FString& Name) { return RobotArticulation->GetActuator(Name); });
}

bool AXArmEnvironment::BuildDictSpaces(FDictSpace& OutObsSpace, FDictSpace& OutActionSpace) const
{
	OutObsSpace.Spaces.Empty();
	OutActionSpace.Spaces.Empty();

	if (!RobotArticulation)
	{
		return false;
	}

	for (const TPair<FString, TObjectPtr<UMjSensor>>& Pair : CachedSensors)
	{
		const UMjSensor* Sensor = Pair.Value.Get();
		if (!Sensor)
		{
			continue;
		}

		const int32 Dim = FMath::Max(Sensor->Dim, 1);
		AddBoxSubSpace(OutObsSpace, Pair.Key, Dim, -ObservationBound, ObservationBound);
	}

	for (const TPair<FString, TObjectPtr<UMjJoint>>& Pair : CachedJoints)
	{
		if (!Pair.Value.Get())
		{
			continue;
		}
		AddBoxSubSpace(OutObsSpace, Pair.Key, 3, -ObservationBound, ObservationBound);
	}

	for (const TPair<FString, TObjectPtr<UMjActuator>>& Pair : CachedActuators)
	{
		const UMjActuator* Act = Pair.Value.Get();
		if (!Act)
		{
			continue;
		}

		FVector2D Range = RobotArticulation->GetActuatorRange(Pair.Key);
		float Low = Range.X;
		float High = Range.Y;
		if (Low == 0.0f && High == 0.0f)
		{
			Low = -ObservationBound;
			High = ObservationBound;
		}

		AddBoxSubSpace(OutActionSpace, Pair.Key, 1, Low, High);
	}

	return (OutObsSpace.Spaces.Num() > 0) && OutActionSpace.Spaces.Num() > 0;
}

void AXArmEnvironment::InitializeEnvironment(TMap<FString, FInteractionDefinition>& OutAgentDefinitions)
{
	OutAgentDefinitions.Empty();
	bDefinitionsInitialized = false;

	if (AgentId.IsEmpty())
	{
		AgentId = TEXT("agent_0");
	}

	if (!RobotArticulation)
	{
		UE_LOG(LogXArmEnv, Error, TEXT("InitializeEnvironment: RobotArticulation is not set on '%s'."), *GetName());
		return;
	}

	if (RobotManager && !RobotManager->IsInitialized())
	{
		UE_LOG(LogXArmEnv, Warning,
			TEXT("InitializeEnvironment: URLab manager '%s' has not compiled yet. Use AURLabScholaGymConnectorManager (or otherwise defer your gym connector's Init) so the articulation's component maps exist before the env caches them."),
			*RobotManager->GetName());
	}

	RebuildComponentCaches();

	if ((CachedSensors.Num() == 0 && CachedJoints.Num() == 0) || CachedActuators.Num() == 0)
	{
		UE_LOG(LogXArmEnv, Error,
			TEXT("InitializeEnvironment: need at least one observation source (sensor or joint) and one actuator (after cache). Actor='%s'."),
			*GetName());
		return;
	}

	FDictSpace ObsSpace;
	FDictSpace ActionSpace;
	if (!BuildDictSpaces(ObsSpace, ActionSpace))
	{
		UE_LOG(LogXArmEnv, Error, TEXT("InitializeEnvironment: failed to build dict spaces for '%s'."), *GetName());
		return;
	}

	FInteractionDefinition Def;
	Def.ObsSpaceDefn = TInstancedStruct<FSpace>::Make<FDictSpace>(ObsSpace);
	Def.ActionSpaceDefn = TInstancedStruct<FSpace>::Make<FDictSpace>(ActionSpace);
	OutAgentDefinitions.Add(AgentId, Def);
	bDefinitionsInitialized = true;

	if (bNormalizeObservations)
	{
		if (FInteractionDefinition* AgentDefinition = OutAgentDefinitions.Find(AgentId))
		{
			if (FDictSpace* ObsSpaceDefn = AgentDefinition->ObsSpaceDefn.GetMutablePtr<FDictSpace>())
			{
				SetBoxSpaceBounds(*ObsSpaceDefn, -ObsClip, ObsClip);
			}
		}
	}

	UE_LOG(LogXArmEnv, Log, TEXT("InitializeEnvironment: agent '%s' obs_keys=%d act_keys=%d."),
		*AgentId, ObsSpace.Spaces.Num(), ActionSpace.Spaces.Num());
}

void AXArmEnvironment::SeedEnvironment(int Seed)
{
	EnvironmentSeed = Seed;
	TargetRandomSeed = Seed;
	TargetRandomEpisodeCounter = 0;
}

void AXArmEnvironment::SetEnvironmentOptions(const TMap<FString, FString>& Options)
{
}

void AXArmEnvironment::ZeroActuatorControls()
{
	for (const TPair<FString, TObjectPtr<UMjActuator>>& Pair : CachedActuators)
	{
		if (UMjActuator* A = Pair.Value.Get())
		{
			A->ResetControl();
		}
	}
}

void AXArmEnvironment::ReadObservations(FDictPoint& OutDict) const
{
	OutDict.Points.Empty();

	if (!RobotArticulation)
	{
		return;
	}

	for (const TPair<FString, TObjectPtr<UMjSensor>>& Pair : CachedSensors)
	{
		TArray<float> Values = RobotArticulation->GetSensorReading(Pair.Key);
		if (Values.Num() == 0)
		{
			if (const UMjSensor* S = Pair.Value.Get())
			{
				Values.Add(S->GetScalarReading());
			}
		}
		OutDict.Points.Add(Pair.Key, TInstancedStruct<FPoint>::Make<FBoxPoint>(FBoxPoint(Values)));
	}

	for (const TPair<FString, TObjectPtr<UMjJoint>>& Pair : CachedJoints)
	{
		const UMjJoint* J = Pair.Value.Get();
		if (!J)
		{
			continue;
		}
		TArray<float> Values;
		Values.Reserve(3);
		Values.Add(J->GetPosition());
		Values.Add(J->GetVelocity());
		Values.Add(J->GetAcceleration());
		OutDict.Points.Add(Pair.Key, TInstancedStruct<FPoint>::Make<FBoxPoint>(FBoxPoint(Values)));
	}
}

bool AXArmEnvironment::ApplyActionDict(const FInstancedStruct& ActionPoint, TMap<FString, float>& OutAppliedScalars)
{
	OutAppliedScalars.Empty();

	if (!ActionPoint.IsValid() || !RobotArticulation)
	{
		return false;
	}

	const FDictPoint* DictPtr = ActionPoint.GetPtr<FDictPoint>();
	if (!DictPtr)
	{
		UE_LOG(LogXArmEnv, Warning, TEXT("Step: action for '%s' is not an FDictPoint."), *AgentId);
		return false;
	}

	for (const TPair<FString, TObjectPtr<UMjActuator>>& ActEntry : CachedActuators)
	{
		const FString& Key = ActEntry.Key;
		const TInstancedStruct<FPoint>* SubPtr = DictPtr->Points.Find(Key);
		if (!SubPtr || !SubPtr->IsValid())
		{
			continue;
		}

		const FBoxPoint* Box = SubPtr->GetPtr<FBoxPoint>();
		if (!Box || Box->Values.Num() < 1)
		{
			UE_LOG(LogXArmEnv, Warning, TEXT("Step: action sub-point for actuator '%s' missing or empty FBoxPoint."), *Key);
			continue;
		}

		const float U = Box->Values[0];
		OutAppliedScalars.Add(Key, U);
		RobotArticulation->SetActuatorControl(Key, U);
	}

	return true;
}

FURLabScholaAgentSnapshot AXArmEnvironment::MakeSnapshot(const FDictPoint& ObsDict,
	const TMap<FString, float>& ActionScalars) const
{
	FURLabScholaAgentSnapshot S;
	S.Robot = RobotArticulation;
	S.Observations.Reserve(ObsDict.Points.Num());
	for (const TPair<FString, TInstancedStruct<FPoint>>& Pair : ObsDict.Points)
	{
		const FBoxPoint* Box = Pair.Value.GetPtr<FBoxPoint>();
		if (!Box)
		{
			continue;
		}

		FURLabScholaNamedSensorReading Row;
		Row.SensorName = Pair.Key;
		Row.Values = Box->Values;
		S.Observations.Add(MoveTemp(Row));
	}
	S.ActionsApplied.Reserve(ActionScalars.Num());
	for (const TPair<FString, float>& Pair : ActionScalars)
	{
		FURLabScholaNamedActuatorCommand Row;
		Row.ActuatorName = Pair.Key;
		Row.Value = Pair.Value;
		S.ActionsApplied.Add(MoveTemp(Row));
	}
	S.GlobalStepIndex = GlobalStepIndex;
	S.EpisodeStepIndex = EpisodeStepIndex;
	S.EnvironmentSeed = EnvironmentSeed;
	return S;
}

FURLabScholaAgentSnapshot AXArmEnvironment::BuildPolicyState(const TMap<FString, float>& ActionScalars,
	TInstancedStruct<FPoint>& OutObservations, TMap<FString, FString>& OutInfo) const
{
	FDictPoint ObsDict;
	ReadObservations(ObsDict);

	const FURLabScholaAgentSnapshot Snapshot = MakeSnapshot(ObsDict, ActionScalars);
	OutInfo = BuildInfo(Snapshot);
	OutObservations = TInstancedStruct<FPoint>::Make<FDictPoint>(ObsDict);
	NormalizePolicyObservations(OutObservations);
	return Snapshot;
}

void AXArmEnvironment::SampleEpisodeTargetLocation()
{
	EpisodeTargetLocationMeters = TargetLocationMeters;
	if (!bRandomizeTargetPerEpisode)
	{
		UpdateTargetWorldCm();
		return;
	}

	const int32 StreamSeed = TargetRandomSeed != 0
		? TargetRandomSeed + TargetRandomEpisodeCounter++
		: GetTypeHash(GetUniqueID()) + TargetRandomEpisodeCounter++;
	TargetRandomStream.Initialize(StreamSeed);

	auto SampleSymmetricOffset = [this](float MaxOffset) -> float
	{
		if (MaxOffset <= 0.f)
		{
			return 0.f;
		}
		return TargetRandomStream.FRandRange(-MaxOffset, MaxOffset);
	};

	auto SamplePositiveOffset = [this](float MaxOffset) -> float
	{
		if (MaxOffset <= 0.f)
		{
			return 0.f;
		}
		return TargetRandomStream.FRandRange(0.f, MaxOffset);
	};

	EpisodeTargetLocationMeters += FVector(
		SampleSymmetricOffset(TargetRandomOffsetMaxMeters.X),
		SampleSymmetricOffset(TargetRandomOffsetMaxMeters.Y),
		SamplePositiveOffset(TargetRandomOffsetMaxMeters.Z));

	UpdateTargetWorldCm();
}

void AXArmEnvironment::UpdateTargetWorldCm()
{
	const FVector LocalCm = LocalMetersToUeCm(EpisodeTargetLocationMeters);
	TargetWorldCm = GetActorTransform().TransformPositionNoScale(LocalCm);
}

void AXArmEnvironment::SyncTargetActorToLocation()
{
	if (!TargetActor)
	{
		return;
	}

	TargetActor->SetActorLocation(TargetWorldCm, false, nullptr, ETeleportType::TeleportPhysics);
}

void AXArmEnvironment::NormalizePolicyObservations(TInstancedStruct<FPoint>& Observations) const
{
	if (!bNormalizeObservations)
	{
		return;
	}

	FDictPoint* Dict = Observations.GetMutablePtr<FDictPoint>();
	if (!Dict)
	{
		return;
	}

	if (TInstancedStruct<FPoint>* TcpPoint = Dict->Points.Find(TcpSensorName))
	{
		if (FBoxPoint* TcpValues = TcpPoint->GetMutablePtr<FBoxPoint>())
		{
			if (TcpValues->Values.Num() >= 3)
			{
				const FVector TcpWorldCm(TcpValues->Values[0], TcpValues->Values[1], TcpValues->Values[2]);
				SetScaledVector(TcpValues->Values, TcpWorldCm - TargetWorldCm, ObsTcpScale * 100.f, ObsClip);
			}
		}
	}

	for (const FString& JointName : ReachObservationJointNames)
	{
		TInstancedStruct<FPoint>* JointPoint = Dict->Points.Find(JointName);
		if (!JointPoint)
		{
			continue;
		}

		FBoxPoint* JointValues = JointPoint->GetMutablePtr<FBoxPoint>();
		if (!JointValues || JointValues->Values.Num() < 3)
		{
			continue;
		}

		JointValues->Values[0] = ScaleAndClip(JointValues->Values[0], ObsJointPosScale, ObsClip);
		JointValues->Values[1] = ScaleAndClip(JointValues->Values[1], ObsJointVelScale, ObsClip);
		JointValues->Values[2] = ScaleAndClip(JointValues->Values[2], ObsJointAccScale, ObsClip);
	}
}

bool AXArmEnvironment::IsSuccessReached(float DistanceMeters, float JointVelocitySquaredSum) const
{
	if (DistanceMeters >= SuccessThreshold)
	{
		return false;
	}

	if (SuccessMaxJointVelocity > 0.f)
	{
		const float MaxVelSq = SuccessMaxJointVelocity * SuccessMaxJointVelocity;
		if (JointVelocitySquaredSum > MaxVelSq)
		{
			return false;
		}
	}

	return true;
}

float AXArmEnvironment::ComputeReward_Implementation(const FURLabScholaAgentSnapshot& Snapshot) const
{
	FReachMetrics Metrics;
	if (!BuildReachMetrics(Snapshot, TcpSensorName, TargetWorldCm, Metrics))
	{
		UE_LOG(LogXArmEnv, Warning, TEXT("ComputeReward: TCP sensor '%s' missing from observations."), *TcpSensorName);
		return StepPenalty;
	}

	float Reward = StepPenalty - (DistanceScale * Metrics.DistanceMeters);

	if (CloseRegionBonusScale > 0.f && CloseRegionMeters > 0.f && Metrics.DistanceMeters < CloseRegionMeters)
	{
		const float Ramp = 1.f - (Metrics.DistanceMeters / CloseRegionMeters);
		Reward += CloseRegionBonusScale * Ramp;
	}

	if (IsSuccessReached(Metrics.DistanceMeters, Metrics.JointVelocitySquaredSum))
	{
		Reward += SuccessBonus;
	}
	else if (bUseOutOfBounds && Metrics.DistanceMeters > OutOfBoundsDistance)
	{
		Reward += OutOfBoundsPenalty;
	}

	if (ActionRatePenaltyScale > 0.f)
	{
		TArray<float> CurrentAction;
		CurrentAction.Reserve(ReachActionActuatorNames.Num());
		for (const FString& ActuatorName : ReachActionActuatorNames)
		{
			float Value = 0.f;
			for (const FURLabScholaNamedActuatorCommand& Cmd : Snapshot.ActionsApplied)
			{
				if (Cmd.ActuatorName == ActuatorName)
				{
					Value = Cmd.Value;
					break;
				}
			}
			CurrentAction.Add(Value);
		}

		if (PreviousActionVector.Num() == CurrentAction.Num())
		{
			float SquaredDelta = 0.f;
			for (int32 i = 0; i < CurrentAction.Num(); ++i)
			{
				const float D = CurrentAction[i] - PreviousActionVector[i];
				SquaredDelta += D * D;
			}
			Reward -= ActionRatePenaltyScale * SquaredDelta;
		}

		PreviousActionVector = MoveTemp(CurrentAction);
	}

	if (JointVelocityPenaltyScale > 0.f)
	{
		Reward -= JointVelocityPenaltyScale * Metrics.JointVelocitySquaredSum;
	}

	return Reward;
}

FURLabScholaTermination AXArmEnvironment::ComputeTermination_Implementation(const FURLabScholaAgentSnapshot& Snapshot) const
{
	FURLabScholaTermination Term;

	FReachMetrics Metrics;
	if (!BuildReachMetrics(Snapshot, TcpSensorName, TargetWorldCm, Metrics))
	{
		return Term;
	}

	if (IsSuccessReached(Metrics.DistanceMeters, Metrics.JointVelocitySquaredSum))
	{
		Term.bTerminated = true;
		return Term;
	}

	if (bUseOutOfBounds && bTerminateOnOutOfBounds && Metrics.DistanceMeters > OutOfBoundsDistance)
	{
		Term.bTerminated = true;
		return Term;
	}

	if (Snapshot.EpisodeStepIndex >= EpisodeStepLimit)
	{
		Term.bTruncated = true;
	}

	return Term;
}

TMap<FString, FString> AXArmEnvironment::BuildInfo_Implementation(const FURLabScholaAgentSnapshot& Snapshot) const
{
	return {};
}

void AXArmEnvironment::Reset(TMap<FString, FInitialAgentState>& OutAgentState)
{
	OutAgentState.Empty();

	if (!bDefinitionsInitialized)
	{
		UE_LOG(LogXArmEnv, Warning, TEXT("Reset called before successful InitializeEnvironment on '%s'."), *GetName());
		return;
	}

	PreviousActionVector.Reset();
	SampleEpisodeTargetLocation();
	SyncTargetActorToLocation();

	ApplyControlSourceIfNeeded();

	if (!RobotArticulation)
	{
		return;
	}

	ZeroActuatorControls();

	if (RobotManager && RobotManager->PhysicsEngine)
	{
		RobotManager->PhysicsEngine->InvalidateStepGate();
	}

	if (!ReachResetKeyframeName.IsEmpty())
	{
		if (RobotManager && RobotManager->PhysicsEngine)
		{
			FScopeLock Lock(&RobotManager->PhysicsEngine->CallbackMutex);
			RobotArticulation->ResetToKeyframe(ReachResetKeyframeName);
		}
		else
		{
			RobotArticulation->ResetToKeyframe(ReachResetKeyframeName);
		}
	}

	EpisodeStepIndex = 0;

	TInstancedStruct<FPoint> Observations;
	TMap<FString, FString> Info;
	const TMap<FString, float> NoActions;
	BuildPolicyState(NoActions, Observations, Info);

	FInitialAgentState Initial;
	Initial.Observations = MoveTemp(Observations);
	Initial.Info = MoveTemp(Info);
	OutAgentState.Add(AgentId, Initial);
}

void AXArmEnvironment::Step(const TMap<FString, FInstancedStruct>& InActions, TMap<FString, FAgentState>& OutAgentStates)
{
	OutAgentStates.Empty();

	checkf(IsInGameThread(),
		TEXT("AXArmEnvironment::Step must run on the game thread. Set bRunEnvironmentsInParallel=false on the GymConnector if needed."));
	if (!IsInGameThread())
	{
		return;
	}

	if (!bDefinitionsInitialized || !RobotArticulation)
	{
		return;
	}

	ApplyControlSourceIfNeeded();

	++GlobalStepIndex;
	++EpisodeStepIndex;

	RobotManager->StepSync(PhysicsStepsPerControlStep);

	TMap<FString, float> AppliedScalars;
	const FInstancedStruct* ActionEntry = InActions.Find(AgentId);
	if (ActionEntry && ActionEntry->IsValid())
	{
		ApplyActionDict(*ActionEntry, AppliedScalars);
	}

	FAgentState State;
	const FURLabScholaAgentSnapshot Snapshot = BuildPolicyState(AppliedScalars, State.Observations, State.Info);
	const float Reward = ComputeReward(Snapshot);
	const FURLabScholaTermination Term = ComputeTermination(Snapshot);

	State.Reward = Reward;
	State.bTerminated = Term.bTerminated;
	State.bTruncated = Term.bTruncated;
	OutAgentStates.Add(AgentId, State);
}
